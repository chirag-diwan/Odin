#include "json_tokeniser.hpp"
#include "definations.hpp"
#include "logging.hpp"
#include <fcntl.h>
#include <pcre2.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

using namespace simdjson;

pcre2_code* compile_regex(const std::string_view& regex){
  int errornumber;
  PCRE2_SIZE erroroffset;

  auto comp_regex = pcre2_compile_8( reinterpret_cast<PCRE2_SPTR>(regex.data()), regex.size(), PCRE2_UTF | PCRE2_UCP, &errornumber, &erroroffset, NULL);

  if (comp_regex == NULL) {
    Log(ERROR , "PCRE2 compilation failed.");
    return nullptr; 
  }

  int jit_result = pcre2_jit_compile( comp_regex, PCRE2_JIT_COMPLETE);

  if (jit_result != 0) {
    Log(WARN, "JIT compilation failed:", jit_result);
  }

  return comp_regex;
}



ODIN_INLINE uint64_t BPETokeniser::getKey(uint32_t first, uint32_t second){
  return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
}

void BPETokeniser::initMaps(simdjson_result<dom::element>& doc){
  if(doc["added_tokens"].has_value()){
    auto added_token = doc["added_tokens"]->get_array();
    size_t added_token_size = added_token.size();
    specialTokens.populate(added_token_size);
  }

  if(doc["model"].has_value() && doc["model"]["vocab"].has_value()){
    auto vocab_obj = doc["model"]["vocab"].get_object();
    size_t vocab_size = vocab_obj->size();
    vocab.populate(vocab_size);
  }

  if(doc["model"].has_value() && doc["model"]["merges"].has_value()){
    auto merges_array = doc["model"]["merges"]->get_array();
    size_t merges_size = merges_array->size();
    merges.populate(merges_size);
  }
}

void BPETokeniser::initPreTokeniser(simdjson_result<dom::element>& doc){
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

void BPETokeniser::fillAddedTokens(simdjson_result<dom::element>& doc){
  if(doc["added_tokens"].has_value()){
    auto added_token = doc["added_tokens"]->get_array();
    size_t fail_count = 0;
    for(auto obj : added_token){
      if(fail_count > 4){
        Log(ERROR , "Fail count exceeded max limit");
        std::exit(-1);
      }
      uint32_t id = obj["id"].get_uint64();
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


void BPETokeniser::fillVocabTokens(simdjson_result<dom::element>& doc){
  if(doc["model"].has_value() && doc["model"]["vocab"].has_value()){
    auto vocab_obj = doc["model"]["vocab"].get_object();
    size_t fail_count = 0;
    for (auto field : vocab_obj) {
      if(fail_count > 4){
        Log(ERROR , "Fail count exceeded max limit");
        std::exit(-1);
      }
      std::string_view key = field.key;
      uint32_t value = field.value.get_uint64();
      if(!vocab.insert(key, value)){
        fail_count ++;
        Log(WARN, "insert into vocab failed");
      }
    }
  }else{
    Log(WARN, "Doc dosen't contains `model` or `[model][vocab]`");
  }

}


void BPETokeniser::fillMergesTokens(simdjson_result<dom::element>& doc){
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
  int fd = open(tokeniser_json.c_str() , O_RDONLY);
  if(fd == -1){
    Log(ERROR , "Cannot open specified file" , tokeniser_json);
    return;
  }

  struct stat sb;
  if(fstat(fd, &sb) == -1){
    Log(ERROR , "Cannot stat specified file" , tokeniser_json);
    return;
  }

  size = sb.st_size;
  file = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE , fd, 0);
  if(file == MAP_FAILED){
    Log(ERROR , "Cannot mmap specified file" , tokeniser_json , "to process memory");
    return;
  }


  auto doc = parserDOM.parse(static_cast<uint8_t*>(file) , size , false);

  initMaps(doc);

  fillAddedTokens(doc);
  initPreTokeniser(doc);
  fillVocabTokens(doc);
  fillMergesTokens(doc);

//#ifdef REGEX
//  specialTokRegex = compile_regex(createSearchRegex());
//  specialTokMatchData =pcre2_match_data_create_from_pattern(specialTokRegex, NULL);
//#endif
  preTokRegex = compile_regex(split_tokeniser.regex);
  preTokMatchData = pcre2_match_data_create_from_pattern(preTokRegex, NULL);

  jitStack = pcre2_jit_stack_create_8(32*1024, 512*1024, nullptr);
  matchContext = pcre2_match_context_create_8(nullptr);
  pcre2_jit_stack_assign_8(matchContext, nullptr , jitStack);


  for(const auto& tok : specialTokens){
    automaton.Insert(tok.first);
  }
  automaton.BuildSufixLinks();
}

void BPETokeniser::tokeniseVec(const std::string& prompt_str , std::vector<uint32_t>& tokens){
  std::string_view prompt = prompt_str;
  specialSeprateTokens.clear();
  PCRE2_SIZE start_offset = 0;
#ifdef REGEX
  while (pcre2_jit_match_8(specialTokRegex, reinterpret_cast<PCRE2_SPTR>(prompt.data()), prompt.size(), start_offset, 0, specialTokMatchData, matchContext) >= 0) {
    PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(specialTokMatchData);
    specialSeprateTokens.emplace_back(prompt.substr(start_offset, ovector[0] - start_offset));
    specialSeprateTokens.emplace_back(prompt.substr(ovector[0], ovector[1] - ovector[0]));
    start_offset = ovector[1]; 
  }

  if(start_offset < prompt.size()){
    specialSeprateTokens.emplace_back(prompt.substr(start_offset, prompt.size() - start_offset));
  }
#else
  std::vector<std::pair<size_t , std::string_view>> out;
  automaton.Search(prompt_str, out);

  size_t last = 0;

  for(const auto&[pos , tok] : out){
    if(pos > last){
      specialSeprateTokens.push_back(prompt.substr(last , pos - last)); 
      last = pos + tok.size();
    }
  }

  if (last < prompt.size()) {
    specialSeprateTokens.push_back( prompt.substr(last));
  }
#endif

  chunks.clear();
  for(const auto& chunk : specialSeprateTokens){
    start_offset = 0;
    if(specialTokens.contains_key(chunk)){
      chunks.push_back(chunk);
      continue;
    }
    while (pcre2_jit_match_8(preTokRegex, reinterpret_cast<PCRE2_SPTR>(chunk.data()), chunk.size(), start_offset, 0, preTokMatchData, matchContext) >= 0) {
      PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(preTokMatchData);

      chunks.push_back(chunk.substr(ovector[0], ovector[1] - ovector[0]));

      start_offset = ovector[1]; 
    }
    if (start_offset < chunk.size()) {
      chunks.push_back(chunk.substr(start_offset));
    }
  }

  for(const auto& chunk : chunks){
    //if(auto val = cache.getValueOf(chunk) ; val.has_value()){
    //  for( const auto b : *val){
    //    tokens.emplace_back(b);
    //  }
    //  continue;
    //}
    if(specialTokens.contains_key(chunk)){
      tokens.emplace_back(*specialTokens.getValueOf(chunk));
      continue;
    }

    bytes.clear();
    for (size_t i = 0; i < chunk.size(); i++) {
      uint8_t raw_byte = static_cast<uint8_t>(chunk[i]);
      auto mapped_str = byteToUnicodeTable[raw_byte].view();

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

        if (!it.has_value()){
          continue;
        }

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

    //if(!cache.insert(chunk, bytes)){
    //  Log(ERROR , "Insert into cache failed");
    //}
    for(const auto b : bytes){
      tokens.emplace_back(b);
    }
  }
}


void BPETokeniser::TokeniseDLL(const std::string& prompt_str , std::vector<uint32_t>& tokens){
  std::string_view prompt = prompt_str;
  specialSeprateTokens.clear();
  PCRE2_SIZE start_offset = 0;
  std::vector<std::pair<size_t , std::string_view>> out;
  automaton.Search(prompt_str, out);
  size_t last = 0;
  for(const auto&[pos , tok] : out){
    if(pos > last){
      specialSeprateTokens.push_back(prompt.substr(last , pos - last)); 
      last = pos + tok.size();
    }
  }
  if (last < prompt.size()) {
    specialSeprateTokens.push_back( prompt.substr(last));
  }

  chunks.clear();
  for(const auto& raw_prompt : specialSeprateTokens){
    start_offset = 0;
    if(specialTokens.contains_key(raw_prompt)){
      chunks.push_back(raw_prompt);
      continue;
    }
    while (pcre2_jit_match_8(preTokRegex, reinterpret_cast<PCRE2_SPTR>(raw_prompt.data()), raw_prompt.size(), start_offset, 0, preTokMatchData, matchContext) >= 0) {
      PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(preTokMatchData);

      chunks.push_back(raw_prompt.substr(ovector[0], ovector[1] - ovector[0]));

      start_offset = ovector[1]; 
    }
    if (start_offset < raw_prompt.size()) {
      chunks.push_back(raw_prompt.substr(start_offset));
    }
  }

  for(const auto& chunk : chunks){
    if(specialTokens.contains_key(chunk)){
      tokens.emplace_back(*specialTokens.getValueOf(chunk));
      continue;
    }

    bytes_dll.clear();
    for (size_t i = 0; i < chunk.size(); i++) {
      uint8_t raw_byte = static_cast<uint8_t>(chunk[i]);
      auto mapped_str = byteToUnicodeTable[raw_byte].view();

      auto id = vocab.getValueOf(mapped_str);
      if(__builtin_expect(!id.has_value(),false)){
        Log(ERROR , "value not found for", mapped_str);
        continue;
      }
      bytes_dll.push(*id);
    }

    while (bytes_dll.size() >= 2) {
      size_t lowest_rank = SIZE_MAX;
      size_t lowest_rank_indx = SIZE_MAX;
      int32_t lowest_rank_indx_prev = -1;
      uint32_t target_merge_id = 0;

      auto current = bytes_dll.first();
      auto next = bytes_dll.next(current);

      while (next != -1) {
        auto key = getKey(bytes_dll[current], bytes_dll[next]);
        auto it  = merges.getValueOf(key);

        if (!it.has_value()){
          current = next;
          next = bytes_dll.next(next);
          continue;
        }

        if ((*it).mergeRank < lowest_rank) {
          lowest_rank = (*it).mergeRank;
          lowest_rank_indx_prev = current;
          lowest_rank_indx = next;
          target_merge_id = (*it).mergeResult;
        }

        current = next;
        next = bytes_dll.next(next);
      }

      if (lowest_rank == SIZE_MAX) {
        break; 
      }

      bytes_dll[lowest_rank_indx_prev] = target_merge_id;
      bytes_dll.erase(lowest_rank_indx);
    }

    for(const auto b : bytes_dll){
      tokens.emplace_back(b.val);
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
  if(file){
    munmap(file, size);
  }

  if(preTokRegex){
    pcre2_code_free(preTokRegex);
  }

#ifdef REGEX
  if(specialTokRegex){
    pcre2_code_free(specialTokRegex);
  }

  if(specialTokMatchData){
    pcre2_match_data_free(specialTokMatchData); 
  }
#endif

  if(preTokMatchData){
    pcre2_match_data_free(preTokMatchData); 
  }

  if(jitStack){
    pcre2_jit_stack_free(jitStack);
  }

  if(matchContext){
    pcre2_match_context_free(matchContext);
  }

}
