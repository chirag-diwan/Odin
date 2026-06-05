#pragma once

#include "bidirectional_map.hpp"
#include "unidirectional_map.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <sys/types.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <utf8proc.h>
#include "logging.hpp"
#include "types.hpp"

#include <string>
#include <vector>



class QwenStyleTokenizer{
  private:
    uint32_t bos_token_id;
    uint32_t eos_token_id;

    bidirectional_map<std::string_view, uint32_t> vocab;
    unidirectional_map<uint64_t , merge_rank_result> merge_priority;

    std::vector<std::string> byte_to_unicode_table;
    uint8_t unicode_to_byte_table[65];

    std::vector<uint32_t> format_block_1;
    std::vector<uint32_t> format_block_2;

    const std::string regex_str ="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    pcre2_code* compiled_regex;

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
    QwenStyleTokenizer(ModelGlobals& globals) :

      bos_token_id(globals.ggml_bos_token_id) ,
      eos_token_id(globals.ggml_eos_token_id) ,
      vocab(globals.token_vocab->size()) ,
      merge_priority(globals.token_merges->size())

      {
        if(globals.token_vocab == nullptr){
          Log(ERROR , "globals.token_vocab is a nullptr");
          return;
        }

        if(globals.token_merges == nullptr){
          Log(ERROR , "globals.token_merges is a nullptr");
          return;
        }



        for(size_t i = 0 ; i < globals.token_vocab->size() ; i++){
          if(__builtin_expect(!vocab.insert(globals.token_vocab->at(i), i) , false)){
            Log(ERROR , "cannot insert key value" , globals.token_vocab->at(i), i);
          }
        }

        for(size_t i = 0 ; i < globals.token_merges->size() ; i++){
          std::string_view merge_pair = globals.token_merges->at(i);
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

          merge_priority.insert(key , { .merge_rank = static_cast<uint32_t>(i) , .merge_result = *merge_result });
        } 



        int errornumber;
        PCRE2_SIZE erroroffset;

        compiled_regex = pcre2_compile(
            reinterpret_cast<PCRE2_SPTR>(regex_str.c_str()),
            PCRE2_ZERO_TERMINATED,
            PCRE2_UTF | PCRE2_UCP, 
            &errornumber,
            &erroroffset,
            NULL
            );

        if (compiled_regex == NULL) {
          Log(ERROR , "PCRE2 compilation failed.");
          return; 
        }


        generate_byte_to_unicode();
        generate_unicode_to_byte();


        format_block_1.push_back(bos_token_id);
        Tokenise("user\n", format_block_1); 

        format_block_2.push_back(eos_token_id); 
        Tokenise("\n", format_block_2);
        format_block_2.push_back(eos_token_id);
        Tokenise("assistant\n", format_block_2);
      }

    void TokeniseFormatted(const std::string& prompt_str , std::vector<uint32_t>& tokens){
      tokens.insert(tokens.end() , format_block_1.begin() , format_block_1.end());
      Tokenise(prompt_str, tokens);
      tokens.insert(tokens.end() , format_block_2.begin() , format_block_2.end());
    }

    void Tokenise(const std::string& prompt_str, std::vector<uint32_t>& tokens){

      std::vector<std::string_view> chunks;
      std::string_view prompt = prompt_str;

      pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(compiled_regex, NULL);
      PCRE2_SIZE start_offset = 0;

      while (pcre2_match(compiled_regex, reinterpret_cast<PCRE2_SPTR>(prompt_str.c_str()), prompt.length(), start_offset, 0, match_data, NULL) >= 0) {
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);

        chunks.push_back(prompt.substr(ovector[0], ovector[1] - ovector[0]));

        start_offset = ovector[1]; 
      }

      pcre2_match_data_free(match_data);

      for(const auto& chunk : chunks){
        std::vector<uint32_t> bytes;

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
            auto it = merge_priority.getValueOf(key);

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

    void Decode(std::vector<uint32_t> tokens){
      for (auto token_id : tokens) {
        auto token_opt = vocab.getKeyOf(token_id);
        if(__builtin_expect(!token_opt.has_value(),false)){
          Log(ERROR , "key not found for ", token_id);
          continue;
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

    ~QwenStyleTokenizer(){
      pcre2_code_free(compiled_regex);
    }
};
