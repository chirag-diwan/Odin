#pragma once

#include "bidirectional_map.hpp"
#include "unidirectional_map.hpp"
#include "span.hpp"
#include "contagious_linked_list.hpp" // Use this for your BPE loop.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <iostream>
#include <vector>
#include <sys/types.h>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <utf8proc.h>
#include "logging.hpp"
#include "types.hpp"

struct ChatMessage {
  std::string role;    // "system", "user", or "assistant"
  std::string content;
};

// Fast Base64 decode table
class LLamaStyleTokenizer {
  private:
    bidirectional_map<std::string_view, uint32_t> vocab;
    bidirectional_map<uint8_t, uint32_t> byte_to_token;
    pcre2_code* compiled_regex;


    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second) {
      return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
    }

  public:
    LLamaStyleTokenizer(ModelGlobals& globals) : vocab(globals.token_vocab->size()), byte_to_token(256) {
      if(globals.token_vocab == nullptr ){
        Log(ERROR, "Globals missing vocab .");
        return;
      }

      if(globals.token_merges == nullptr){
        Log(ERROR, "Globals missing merges.");
        return;
      }

      for (size_t i = 0; i < globals.token_vocab->size(); i++) {
        std::string_view token = globals.token_vocab->at(i); 

        vocab.insert(token, i);

        if(token.size() == 1){
          uint8_t raw_byte = static_cast<uint8_t>(token[0]);
          byte_to_token.insert(raw_byte, i); 
        }
      }

      std::string regex_str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]? \\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
      int errornumber;
      PCRE2_SIZE erroroffset;
      compiled_regex = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex_str.c_str()), PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &errornumber, &erroroffset, NULL);
    }

    void Tokenise(std::string prompt_str, std::vector<uint32_t>& tokens) {
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

      c_list<2048> bytes;

      for (const auto& chunk : chunks) {
        bytes.clear(); 

        for (size_t i = 0; i < chunk.size(); i++) {
          uint8_t raw_byte = static_cast<uint8_t>(chunk[i]);
          if(!byte_to_token.contains_key(raw_byte))continue;
          uint32_t base_token_id = *byte_to_token.getValueOf(raw_byte);
          bytes.push(base_token_id);
        }

        while (bytes.size() >= 2) {
          size_t lowest_rank = SIZE_MAX;
          size_t lowest_rank_indx = SIZE_MAX;
          uint32_t target_merge_id = 0;

          for (size_t i = 1; i < bytes.size(); i++) {
            auto prev = bytes.at(i - 1);
            auto curr = bytes.at(i);

            if(prev == nullptr) continue;
            if(curr == nullptr) continue;

            if(!vocab.contains_value(prev->val_) || !vocab.contains_value(curr->val_)) continue;

            auto prev_str_view = *vocab.getKeyOf(prev->val_);
            auto curr_str_view = *vocab.getKeyOf(curr->val_);

            std::string merge;
            merge.reserve(prev_str_view.size() + curr_str_view.size());
            merge.append(prev_str_view);
            merge.append(curr_str_view);

            std::string_view merge_view = merge;

            if(!vocab.contains_key(merge_view))continue;

            auto merge_id = *vocab.getValueOf(merge_view);
            if(merge_id < lowest_rank){
              lowest_rank_indx = i;
              lowest_rank = merge_id;
              target_merge_id = merge_id;
            }
          }

          if (lowest_rank == SIZE_MAX) {
            break; 
          }

          bytes.at(lowest_rank_indx - 1)->val_ = target_merge_id;
          bytes.erase(lowest_rank_indx);
        }
        for(const auto byte : bytes){
          tokens.push_back(byte.val_);
        }
      }
    }

    void Decode(span<uint32_t> tokens) {
      for (auto token_id : tokens) {
        auto token_opt = vocab.getKeyOf(token_id);
        if(!token_opt.has_value()) continue;

        auto token_str = *token_opt;
        std::cout.write(token_str.data(), token_str.size());
        std::fflush(stdout); 
      }
    }

    void Decode(uint32_t token_id) { // Fixed signature mismatch
      auto token_opt = vocab.getKeyOf(token_id);
      if(!token_opt.has_value()) return;

      auto token_str = *token_opt;
      std::cout.write(token_str.data(), token_str.size());
      std::fflush(stdout); 
    }

    void TokeniseChatFormat(const std::vector<ChatMessage>& messages, std::vector<uint32_t>& tokens) {
      auto bos_opt = vocab.getValueOf("<|begin_of_text|>");
      auto start_header_opt = vocab.getValueOf("<|start_header_id|>");
      auto end_header_opt = vocab.getValueOf("<|end_header_id|>");
      auto eot_opt = vocab.getValueOf("<|eot_id|>");

      if (!bos_opt.has_value() || !start_header_opt.has_value() || 
          !end_header_opt.has_value() || !eot_opt.has_value()) {
        Log(ERROR, "Missing LLaMA 3 structural tokens in vocabulary.");
        return;
      }

      tokens.push_back(*bos_opt);

      for (const auto& msg : messages) {
        // Open Header
        tokens.push_back(*start_header_opt);

        // Tokenise the role (usually "system", "user", or "assistant")
        Tokenise(msg.role, tokens);

        // Close Header
        tokens.push_back(*end_header_opt);

        // LLaMA 3 strictly requires two newlines after the header.
        // These are standard text and MUST go through BPE.
        Tokenise("\n\n", tokens);

        // Tokenise the actual payload
        Tokenise(msg.content, tokens);

        // End of Turn (Signals the model this message is over)
        tokens.push_back(*eot_opt);
      }

      // 4. Generation Prompt
      // You must set up the sequence so the model knows it is its turn to speak.
      tokens.push_back(*start_header_opt);
      Tokenise("assistant", tokens);
      tokens.push_back(*end_header_opt);
      Tokenise("\n\n", tokens);
    }

    ~LLamaStyleTokenizer() {
      if(compiled_regex) pcre2_code_free(compiled_regex);
    }
};
