#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <unordered_map>
#include <vector>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <utf8proc.h>
#include "logging.hpp"
#include "types.hpp"

#include <string>
#include <vector>

std::vector<std::string> generate_byte_to_unicode() {
  std::vector<std::string> byte_to_unicode(256);
  int n = 0;
  for (int b = 0; b < 256; b++) {
    // Range of printable characters that map to themselves
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
      byte_to_unicode[b] = std::string(1, static_cast<char>(b));
    } else {
      // Map to U+0100 and above
      int unicode_val = 256 + n;
      std::string utf8_char;
      // Convert to 2-byte UTF-8 (since range is 256-320)
      utf8_char.push_back(static_cast<char>(0xC0 | (unicode_val >> 6)));
      utf8_char.push_back(static_cast<char>(0x80 | (unicode_val & 0x3F)));

      byte_to_unicode[b] = utf8_char;
      n++;
    }
  }
  return byte_to_unicode;
}

class Tokeniser{
  private:
    std::unordered_map<std::string_view, uint64_t> vocab;
    std::unordered_map<uint64_t , std::string_view> tokens_to_string;
    std::unordered_map<uint64_t, MergeRV> merge_priority;
    std::vector<std::string> byte_to_unicode_table;


    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second){
      return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
    }

  public:
    Tokeniser(MetadataKV_t& metadata_key_values){
      for(const auto& kv : metadata_key_values){
        if(kv.name == "tokenizer.ggml.tokens"){
          for(size_t i = 0 ; i <kv.value.array.strings.size() ; i++){
            vocab[kv.value.array.strings[i]] = i;
            tokens_to_string[i] = kv.value.array.strings[i];
          }
        }else if(kv.name == "tokenizer.ggml.merges"){
          for(size_t i = 0 ; i <kv.value.array.strings.size() ; i++){
            std::string_view merge_pair = kv.value.array.strings[i];
            auto split_point = merge_pair.find(' ');
            std::string_view first = merge_pair.substr(0 , split_point);
            std::string_view second = merge_pair.substr(split_point + 1);
            auto first_idx = vocab.at(first);
            auto second_idx = vocab.at(second);

            auto key = getKey(first_idx , second_idx);
            std::string result;
            result.reserve(first.size() + second.size());

            result.append(first);
            result.append(second);

            merge_priority[key] = {
              .merge_rank = i ,
              .merge_result = vocab.at(result)
            };
          }
        }
      }
      byte_to_unicode_table = generate_byte_to_unicode();
    }

    void Tokenise(std::string prompt, std::vector<int32_t>& tokens){

      const std::string regex_str ="(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
      std::vector<std::string> chunks;

      int errornumber;
      PCRE2_SIZE erroroffset;

      pcre2_code* re = pcre2_compile(
          reinterpret_cast<PCRE2_SPTR>(regex_str.c_str()),
          PCRE2_ZERO_TERMINATED,
          PCRE2_UTF | PCRE2_UCP, 
          &errornumber,
          &erroroffset,
          NULL
          );

      if (re == NULL) {
        Log(ERROR , "PCRE2 compilation failed.");
        return; 
      }

      pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);
      PCRE2_SIZE start_offset = 0;

      while (pcre2_match(re, reinterpret_cast<PCRE2_SPTR>(prompt.c_str()), prompt.length(), start_offset, 0, match_data, NULL) >= 0) {
        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);

        chunks.push_back(prompt.substr(ovector[0], ovector[1] - ovector[0]));

        start_offset = ovector[1]; 
      }

      pcre2_match_data_free(match_data);
      pcre2_code_free(re);


      std::vector<std::string> modified_chunk;

      for(const auto& chunk : chunks){
        modified_chunk.emplace_back(chunk);
      }
      modified_chunk.emplace_back( "assistant\n");



      tokens.emplace_back(151644);
      for(const auto& chunk : modified_chunk){
        std::vector<uint32_t> bytes;

        for (size_t i = 0; i < chunk.size(); i++) {
          uint8_t raw_byte = static_cast<uint8_t>(chunk[i]);
          auto mapped_str = byte_to_unicode_table[raw_byte];

          try {
            auto id = vocab.at(mapped_str);
            bytes.emplace_back(id);
          } catch (std::out_of_range& e) {
            Log(ERROR, "Not found in vocab: ", mapped_str);
          }

        }

        while (bytes.size() >= 2) {
          size_t lowest_rank = SIZE_MAX;
          size_t lowest_rank_indx = SIZE_MAX;
          uint32_t target_merge_id = 0;

          for (size_t i = 1; i < bytes.size(); i++) {
            auto key = getKey(bytes[i - 1], bytes[i]);
            auto it = merge_priority.find(key);

            if (it == merge_priority.end()) continue;

            if (it->second.merge_rank < lowest_rank) {
              lowest_rank = it->second.merge_rank;
              lowest_rank_indx = i;
              target_merge_id = it->second.merge_result;
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
      tokens.emplace_back(151645);
      tokens.emplace_back(151643);
    }

    void Decode(std::vector<int32_t> tokens){
      for(auto token : tokens){
        Log(tokens_to_string[token]);
      }
    }
};
