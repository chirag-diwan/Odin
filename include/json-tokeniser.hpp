#pragma once

#include "logging.hpp"
#include "types.hpp"
#include "unidirectional_map.hpp"
#include "bidirectional_map.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <string>
#include <simdjson.h>
#include <string_view>
#include <type_traits>

using namespace simdjson;

pcre2_code* compile_regex(const std::string_view& regex){
  int errornumber;
  PCRE2_SIZE erroroffset;

  auto regex_str = std::string(regex.data() , regex.size());

  auto comp_regex = pcre2_compile_8(
      reinterpret_cast<PCRE2_SPTR>(regex_str.c_str()),
      PCRE2_ZERO_TERMINATED,
      PCRE2_UTF | PCRE2_UCP, 
      &errornumber,
      &erroroffset,
      NULL
      );

  if (comp_regex == NULL) {
    Log(ERROR , "PCRE2 compilation failed.");
    return nullptr; 
  }
  return comp_regex;
}

class BPETokeniser{
  protected:
    TokeniserConfig config;
    PreTokeniser split_tokeniser;

    ondemand::parser parser;
    const padded_string json;

    bidirectional_map<std::string_view, uint32_t> vocab;
    bidirectional_map<std::string_view, uint32_t> special_tokens;
    unidirectional_map<uint64_t, merge_rank_result> merges;

    pcre2_code* pre_tok_regex;
    pcre2_code* special_tok_regex;


    std::vector<std::string> byte_to_unicode_table;
    uint8_t unicode_to_byte_table[65];


    // XXX created by llm
    void generate_unicode_to_byte(){
      int n = 0;
      for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
        } else {
          int unicode_val = 256 + n;
          unicode_to_byte_table[unicode_val - 256] = static_cast<uint8_t>(b);
          n++;
        }
      }
    }

    // XXX created by llm
    void generate_byte_to_unicode() {
      byte_to_unicode_table.resize(256);
      int n = 0;
      for (int b = 0; b < 256; b++) {
        // Range of printable characters that map to themselves
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
          byte_to_unicode_table[b] = std::string(1, static_cast<char>(b));
        } else {
          // Map to U+0100 and above
          int unicode_val = 256 + n;
          std::string utf8_char;
          // Convert to 2-byte UTF-8 (since range is 256-320)
          utf8_char.push_back(static_cast<char>(0xC0 | (unicode_val >> 6)));
          utf8_char.push_back(static_cast<char>(0x80 | (unicode_val & 0x3F)));

          byte_to_unicode_table[b] = utf8_char;
          n++;
        }
      }
    }


    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second){
      return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
    }

  public:
    BPETokeniser(const std::string& tokeniser_json) : json(padded_string::load(tokeniser_json)) {
      auto doc = parser.iterate(json);

      auto added_token = doc["added_tokens"]->get_array();
      size_t added_token_size = added_token->count_elements();
      special_tokens = bidirectional_map<std::string_view, uint32_t>(added_token_size);

      auto vocab_obj = doc["model"]["vocab"].get_object();
      size_t vocab_size = vocab_obj->count_fields();
      vocab = bidirectional_map<std::string_view, uint32_t>(vocab_size);


      auto merges_array = doc["model"]["merges"]->get_array();
      size_t merges_size = merges_array->count_elements();
      merges = unidirectional_map<uint64_t , merge_rank_result>(merges_size);

      //Re iterate
      auto doc_reinit = parser.iterate(json);

      added_token = doc_reinit["added_tokens"]->get_array();
      for(auto obj : added_token){
        uint32_t id = obj["id"]->get_uint32();
        std::string_view token = obj["content"]->get_string();
        special_tokens.insert(token, id);
      }

      auto pretokenizers = doc_reinit["pre_tokenizer"]["pretokenizers"];
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


      vocab_obj = doc_reinit["model"]["vocab"].get_object();
      for (auto field : vocab_obj) {
        std::string_view key = field->unescaped_key();
        uint32_t value = uint32_t(field.value());
        vocab.insert(key, value);
      }


      merges_array = doc_reinit["model"]["merges"]->get_array();
      size_t i = 0;
      for(auto element : merges_array){
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
        merges.insert(key , { .merge_rank = static_cast<uint32_t>(i) , .merge_result = *merge_result });
        i++;
      }

      pre_tok_regex = compile_regex(split_tokeniser.regex);

      std::string special_tokens_str;
      special_tokens_str.reserve(special_tokens.size()*10);

      special_tokens_str.append("(?:");
      for(const auto& special_tok_id_pair : special_tokens){
        special_tokens_str += "\\Q";
        special_tokens_str.append(special_tok_id_pair.first.data() , special_tok_id_pair.first.size());
        special_tokens_str += "\\E|";
      }
      special_tokens_str.pop_back();
      special_tokens_str.push_back(')');


      special_tok_regex = compile_regex(special_tokens_str);


      generate_byte_to_unicode();
      generate_unicode_to_byte();
    }

    void Tokenise(const std::string& prompt_str , std::vector<uint32_t>& tokens){
      std::string_view prompt = prompt_str;

      std::vector<std::string_view> special_seprate_tokens;

      pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(special_tok_regex ,NULL);
      PCRE2_SIZE start_offset = 0;


      while (pcre2_match(special_tok_regex, reinterpret_cast<PCRE2_SPTR>(prompt_str.c_str()), prompt.length(), start_offset, 0, match_data, NULL) >= 0) {
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
        special_seprate_tokens.push_back(prompt.substr(ovector[0], ovector[1] - ovector[0]));
        start_offset = ovector[1]; 
      }

      pcre2_match_data_free(match_data);

      for(const auto& foo : special_seprate_tokens){
        Log(foo);
      }

      match_data = pcre2_match_data_create_from_pattern(pre_tok_regex, NULL);
      start_offset = 0;

      std::vector<std::string_view> chunks;
      for(const auto& raw_prompt : special_seprate_tokens){
        if(special_tokens.contains_key(raw_prompt)){
          chunks.push_back(raw_prompt);
          continue;
        }
        while (pcre2_match(pre_tok_regex, reinterpret_cast<PCRE2_SPTR>(raw_prompt.data()), raw_prompt.length(), start_offset, 0, match_data, NULL) >= 0) {
          PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);

          chunks.push_back(raw_prompt.substr(ovector[0], ovector[1] - ovector[0]));

          start_offset = ovector[1]; 
        }
      }

      pcre2_match_data_free(match_data);

      for(const auto& chunk : chunks){
        if(special_tokens.contains_key(chunk)){
          tokens.emplace_back(*special_tokens.getValueOf(chunk));
          continue;
        }
        std::vector<uint32_t> bytes;
        bytes.reserve(chunk.size());

        for (size_t i = 0; i < chunk.size(); i++) {
          uint8_t raw_byte = static_cast<uint8_t>(chunk[i]);
          auto mapped_str = byte_to_unicode_table[raw_byte];

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

            if ((*it).merge_rank < lowest_rank) {
              lowest_rank = (*it).merge_rank;
              lowest_rank_indx = i;
              target_merge_id = (*it).merge_result;
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

    void Decode(uint32_t token_id){
      auto token_opt = vocab.getKeyOf(token_id);
      if(__builtin_expect(!token_opt.has_value(),false)){
        Log(ERROR , "key not found for ", token_id);
        return;
      }

      auto token_str = *token_opt;
      for (size_t i = 0; i < token_str.size(); ) {
        unsigned char c = token_str[i];

        if ((c & 0x80) == 0) {
          std::putchar(c); 
          i++;
        } 
        else if ((c & 0xE0) == 0xC0) {
          unsigned char c2 = token_str[i + 1];
          uint16_t unicode_val = ((c & 0x1F) << 6) | (c2 & 0x3F);

          uint8_t original_byte = unicode_to_byte_table[unicode_val - 256];
          std::putchar(original_byte);

          i += 2; 
        } 
        else {
          Log(ERROR, "Malformed BPE sequence detected.");
          break;
        }
      }
      std::fflush(stdout); 
    }

    ~BPETokeniser(){
      pcre2_code_free(pre_tok_regex);
    }
};
