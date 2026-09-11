#include "json_tokeniser.hpp"
#include <sys/types.h>

using namespace simdjson;

pcre2_code* compile_regex(const std::string_view& regex){
  int errornumber;
  PCRE2_SIZE erroroffset;

  auto comp_regex = pcre2_compile_8(
      reinterpret_cast<PCRE2_SPTR>(regex.data()),
      regex.size(),
      PCRE2_UTF | PCRE2_UCP, 
      &errornumber,
      &erroroffset,
      NULL
      );

  if (comp_regex == NULL) {
    Log(ERROR , "PCRE2 compilation failed.");
    return nullptr; 
  }

  int jit_result = pcre2_jit_compile(
      comp_regex,
      PCRE2_JIT_COMPLETE
      );

  if (jit_result != 0) {
    Log(WARN, "JIT compilation failed:", jit_result);
  }

  return comp_regex;

  return comp_regex;
}




void BPETokeniser::generateUnicodeToByte(){
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
    } else {
      int unicode_val = 256 + n;
      unicodeToByteTable[unicode_val - 256] = static_cast<uint8_t>(b);
      n++;
    }
  }
}

// XXX created by llm
void BPETokeniser::generateByteToUnicode() {
  byteToUnicodeTable.resize(256);
  int n = 0;
  for (int b = 0; b < 256; b++) {
    // Range of printable characters that map to themselves
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
      byteToUnicodeTable[b] = std::string(1, static_cast<char>(b));
    } else {
      // Map to U+0100 and above
      int unicode_val = 256 + n;
      std::string utf8_char;
      // Convert to 2-byte UTF-8 (since range is 256-320)
      utf8_char.push_back(static_cast<char>(0xC0 | (unicode_val >> 6)));
      utf8_char.push_back(static_cast<char>(0x80 | (unicode_val & 0x3F)));

      byteToUnicodeTable[b] = utf8_char;
      n++;
    }
  }
}


__attribute__((always_inline)) inline uint64_t BPETokeniser::getKey(uint32_t first, uint32_t second){
  return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
}

void BPETokeniser::initMaps(simdjson_result<ondemand::document>& doc){
  if(doc["added_tokens"].has_value()){
    auto added_token = doc["added_tokens"]->get_array();
    size_t added_token_size = added_token->count_elements();
    specialTokens.populate(added_token_size);
  }

  if(doc["model"].has_value() && doc["model"]["vocab"].has_value()){
    auto vocab_obj = doc["model"]["vocab"].get_object();
    size_t vocab_size = vocab_obj->count_fields();
    vocab.populate(vocab_size);
  }

  if(doc["model"].has_value() && doc["model"]["merges"].has_value()){
    auto merges_array = doc["model"]["merges"]->get_array();
    size_t merges_size = merges_array->count_elements();
    merges.populate(merges_size);
  }
}

void BPETokeniser::initPreTokeniser(simdjson_result<ondemand::document>& doc){
  if(doc["pre_tokenizer"].has_value() && doc["pre_tokenizer"]["pretokenizers"].has_value()){
    auto pretokenizers = doc["pre_tokenizer"]["pretokenizers"];
    for(auto obj : pretokenizers){
      std::string_view type = obj["type"]->get_string();
      if(type == "Split"){
        split_tokeniser.regex = obj["pattern"]["Regex"]->get_string();
        split_tokeniser.behavior = obj["behavior"]->get_string();
        split_tokeniser.invert = obj["invert"]->get_bool();
        break;
      }else{
        Log(ERROR , "PreTokeniser type Split not found");
      }
    }
  }
}

void BPETokeniser::fillAddedTokens(simdjson_result<ondemand::document>& doc){
  if(doc["added_tokens"].has_value()){
    auto added_token = doc["added_tokens"]->get_array();
    size_t fail_count = 0;
    for(auto obj : added_token){
      if(fail_count > 4){
        Log(ERROR , "Fail count exceeded max limit");
        std::exit(-1);
      }
      uint32_t id = obj["id"]->get_uint32();
      std::string_view token = obj["content"]->get_string();
      if(!specialTokens.insert(token, id)){
        Log(WARN , "insert into specialTokens failed");
        fail_count++;
      }
    }
  }else{
    Log(WARN, "Doc dosen't contains added_tokens");
  }
}


void BPETokeniser::fillVocabTokens(simdjson_result<ondemand::document>& doc){
  if(doc["model"].has_value() && doc["model"]["vocab"].has_value()){
    auto vocab_obj = doc["model"]["vocab"].get_object();
    size_t fail_count = 0;
    for (auto field : vocab_obj) {
      if(fail_count > 4){
        Log(ERROR , "Fail count exceeded max limit");
        std::exit(-1);
      }
      std::string_view key = field->unescaped_key();
      uint32_t value = uint32_t(field.value());
      if(!vocab.insert(key, value)){
        fail_count ++;
        Log(WARN, "insert into vocab failed");
      }
    }
  }else{
    Log(WARN, "Doc dosen't contains `model` or `[model][vocab]`");
  }

}


void BPETokeniser::fillMergesTokens(simdjson_result<ondemand::document>& doc){
  if(doc["model"].has_value() && doc["model"]["merges"].has_value()){
    auto merges_array = doc["model"]["merges"]->get_array();
    size_t i = 0;
    size_t fail_count = 0;
    for(auto element : merges_array){
      if(fail_count > 4){
        Log(ERROR , "Fail count exceeded max limit");
        std::exit(-1);
      }
      std::string_view merge_pair = element.get_string();
      auto split_point = merge_pair.find(' ');
      std::string_view first = merge_pair.substr(0 , split_point);
      std::string_view second = merge_pair.substr(split_point + 1);
      auto first_idx = vocab.getValueOf(first);
      auto second_idx = vocab.getValueOf(second);

      if(__builtin_expect(!first_idx.has_value(),false)){
        Log(ERROR , "value not found for key" , first);
        continue;
      }
      if (__builtin_expect(!second_idx.has_value(),false)) {
        Log(ERROR , "value not found for key" , second);
        continue;
      }


      auto key = getKey(*first_idx, *second_idx);
      std::string result;
      result.reserve(first.size() + second.size());

      result.append(first);
      result.append(second);

      auto merge_result = vocab.getValueOf(result);
      if(__builtin_expect(!merge_result.has_value(),false)){
        Log(ERROR , "value not found for key" , result);
        continue;
      }

      if(!merges.insert(key , { .mergeRank= static_cast<uint32_t>(i) , .mergeResult = *merge_result })){
        fail_count ++;
        Log(WARN, "insert into merges failed");
      }
      i++;
    }
  }else{
    Log(WARN, "Doc dosen't contains `model` or `[model][merges]`");
  }
}

std::string BPETokeniser::createSearchRegex(){
  std::string special_tokens_str;
  special_tokens_str.reserve(specialTokens.size()*10);

  special_tokens_str.append("(?:");
  for(const auto& [tok , _] : specialTokens){
    special_tokens_str += "\\Q";
    special_tokens_str.append(tok.data() , tok.size());
    special_tokens_str += "\\E|";
  }
  special_tokens_str.pop_back();
  special_tokens_str.push_back(')');
  return special_tokens_str;
}

void BPETokeniser::Open(const std::string& tokeniser_json){
  json = padded_string::load(tokeniser_json);
  auto doc = parser.iterate(json);
  initMaps(doc);

  auto doc_reinit = parser.iterate(json);

  fillAddedTokens(doc_reinit);
  initPreTokeniser(doc_reinit);
  fillVocabTokens(doc_reinit);
  fillMergesTokens(doc_reinit);

  preTokRegex = compile_regex(split_tokeniser.regex);
  specialTokRegex = compile_regex(createSearchRegex());

  generateByteToUnicode();
  generateUnicodeToByte();

  jitStack = pcre2_jit_stack_create_8(32*1024, 512*1024, nullptr);
  matchContext = pcre2_match_context_create_8(nullptr);
  pcre2_jit_stack_assign_8(matchContext, nullptr , jitStack);
}

void BPETokeniser::Tokenise(const std::string& prompt_str , std::vector<uint32_t>& tokens){
  std::string_view prompt = prompt_str;

  specialSeprateTokens.clear();

  pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(specialTokRegex ,NULL);
  PCRE2_SIZE start_offset = 0;

  while (pcre2_jit_match_8(specialTokRegex, reinterpret_cast<PCRE2_SPTR>(prompt_str.c_str()), prompt.size(), start_offset, 0, match_data, matchContext) >= 0) {
    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
    specialSeprateTokens.emplace_back(prompt.substr(start_offset, ovector[0] - start_offset));
    specialSeprateTokens.emplace_back(prompt.substr(ovector[0], ovector[1] - ovector[0]));
    start_offset = ovector[1]; 
  }

  if(start_offset < prompt.size()){
    specialSeprateTokens.emplace_back(prompt.substr(start_offset, prompt.size() - start_offset));
  }

  pcre2_match_data_free(match_data);

  match_data = pcre2_match_data_create_from_pattern(preTokRegex, NULL);


  chunks.clear();
  for(const auto& raw_prompt : specialSeprateTokens){
    start_offset = 0;
    if(specialTokens.contains_key(raw_prompt)){
      chunks.push_back(raw_prompt);
      continue;
    }
    while (pcre2_jit_match_8(preTokRegex, reinterpret_cast<PCRE2_SPTR>(raw_prompt.data()), raw_prompt.size(), start_offset, 0, match_data, matchContext) >= 0) {
      PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);

      chunks.push_back(raw_prompt.substr(ovector[0], ovector[1] - ovector[0]));

      start_offset = ovector[1]; 
    }
  }

  pcre2_match_data_free(match_data);

  for(const auto& chunk : chunks){
    if(specialTokens.contains_key(chunk)){
      tokens.emplace_back(*specialTokens.getValueOf(chunk));
      continue;
    }

    bytes.clear();

    for (size_t i = 0; i < chunk.size(); i++) {
      uint8_t raw_byte = static_cast<uint8_t>(chunk[i]);
      auto mapped_str = byteToUnicodeTable[raw_byte];

      auto id = vocab.getValueOf(mapped_str);
      if(__builtin_expect(!id.has_value(),false)){
        Log(ERROR , "value not found for", mapped_str);
        continue;
      }
      bytes.emplace_back(*id);
    }

    while (bytes.size() >= 2) {
      size_t lowest_rank = SIZE_MAX;
      size_t lowest_rank_indx = SIZE_MAX;
      uint32_t target_merge_id = 0;

      for (size_t i = 1; i < bytes.size(); i++) {
        auto key = getKey(bytes[i - 1], bytes[i]);
        auto it = merges.getValueOf(key);

        if (!it.has_value()) continue;

        if ((*it).mergeRank < lowest_rank) {
          lowest_rank = (*it).mergeRank;
          lowest_rank_indx = i;
          target_merge_id = (*it).mergeResult;
        }
      }

      if (lowest_rank == SIZE_MAX) {
        break; 
      }

      bytes[lowest_rank_indx - 1] = target_merge_id;
      bytes.erase(bytes.begin() + lowest_rank_indx);
    }
    for(const auto b : bytes){
      tokens.emplace_back(b);
    }
  }
}

std::optional<std::string> BPETokeniser::Decode(uint32_t token_id){
  auto token_opt = vocab.getKeyOf(token_id);

  if(__builtin_expect(!token_opt.has_value(),false)){
    token_opt = specialTokens.getKeyOf(token_id);
    if(!token_opt.has_value()){
      return std::nullopt;
    }
  }

  auto token_str = *token_opt;
  std::string token = "";
  token.reserve(token_str.size());
  for (size_t i = 0; i < token_str.size(); ) {
    unsigned char c = token_str[i];

    if ((c & 0x80) == 0) {
      token.push_back(c); 
      i++;
    } 
    else if ((c & 0xE0) == 0xC0) {
      unsigned char c2 = token_str[i + 1];
      uint16_t unicode_val = ((c & 0x1F) << 6) | (c2 & 0x3F);

      uint8_t original_byte = unicodeToByteTable[unicode_val - 256];
      token.push_back(original_byte);

      i += 2; 
    } 
    else {
      Log(ERROR, "Malformed BPE sequence detected.");
      break;
    }
  }

  return token;
}

void BPETokeniser::Delete(){
  pcre2_code_free(preTokRegex);
}
