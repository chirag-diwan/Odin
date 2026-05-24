#include "bidirectional_map.hpp"
#include "errors.hpp"
#include "unidirectional_map.hpp"
#include "span.hpp"
#include "simdjson.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
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

// XXX created by llm
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
    bidirectional_map<std::string, uint32_t> vocab;
    unidirectional_map<MergeRV> merge_priority;
    std::vector<std::string> byte_to_unicode_table;
    uint8_t unicode_to_byte_table[65];

    pcre2_code* compiled_regex;

    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second){
      return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
    }

  public:
    Tokeniser(std::string& tokeniser_path){
      std::ifstream tokeniser_json(tokeniser_path);
      std::string json_content(std::istreambuf_iterator<char>{tokeniser_json} , std::istreambuf_iterator<char>{});
      tokeniser_json.close();

      simdjson::dom::parser parser;
      simdjson::dom::element doc = parser.parse(json_content.data() , json_content.size() , false);
      if(doc.at_key("added_tokens")->is_array()){
        for(const auto& element : doc.at_key("added_tokens")->get_array()){
          uint32_t id = element.at_key("id")->get_uint64();
          auto content = element.at_key("content")->get_string();
          vocab.insert(std::string(content->data() , content->size()), id);
        }
      }
      auto pre_tokenizer_obj = doc.at_key("pre_tokenizer");
      auto pre_tokenizers_array = pre_tokenizer_obj->at_key("pretokenizers");
      std::string regex;
      Errorif(!pre_tokenizers_array->is_array(), "Pre tokenizers is not any array");
      for(const auto& element : pre_tokenizers_array){
        if(element.at_key("type")->get_string()->compare("Split") == 0){
          auto json_regex_string = element.at_key("pattern")->at_key("Regex")->get_string();
          regex = std::string(json_regex_string->data() , json_regex_string->size());
        }
      }
    }

    void Tokenise(std::string prompt_str, std::vector<uint32_t>& tokens){
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


    void Decode(span<uint32_t> tokens){
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


    void Decode(int32_t token_id){
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

    ~Tokeniser(){
      pcre2_code_free(compiled_regex);


    }
};
