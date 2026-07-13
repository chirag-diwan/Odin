#pragma once

#include "./types.hpp"
#include "./data_structures/unidirectional_map.hpp"
#include "./data_structures/bidirectional_map.hpp"
#include <cstdint>
#include <optional>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <string>
#include "../external/simdjson/simdjson.h"
#include <string_view>

using namespace simdjson;

pcre2_code* compile_regex(const std::string_view& regex);

class BPETokeniser{
  private:
    std::vector<std::string_view> chunks;
    std::vector<uint32_t> bytes;


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

    pcre2_jit_stack* jit_stack;
    pcre2_match_context* match_context;

    std::vector<std::string> byte_to_unicode_table;
    uint8_t unicode_to_byte_table[65];


    // XXX created by llm
    void generate_unicode_to_byte();

    // XXX created by llm
    void generate_byte_to_unicode() ;

    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second);

    void init_maps(simdjson_result<ondemand::document>& doc);

    void init_pre_tokeniser(simdjson_result<ondemand::document>& doc);

    void fill_added_tokens(simdjson_result<ondemand::document>& doc);

    void fill_vocab_tokens(simdjson_result<ondemand::document>& doc);

    void fill_merges_tokens(simdjson_result<ondemand::document>& doc);

    std::string create_search_regex();

  public:

    std::vector<std::string_view> special_seprate_tokens;

    BPETokeniser(const std::string& tokeniser_json);

    void Tokenise(const std::string& prompt_str , std::vector<uint32_t>& tokens);
    std::optional<std::string> Decode(uint32_t token_id);

    ~BPETokeniser();
};
