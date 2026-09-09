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
    padded_string json;

    bidirectional_map<std::string_view, uint32_t> vocab;
    bidirectional_map<std::string_view, uint32_t> specialTokens;
    unidirectional_map<uint64_t, merge_rank_result> merges;

    pcre2_code* preTokRegex;
    pcre2_code* specialTokRegex;

    pcre2_jit_stack* jitStack;
    pcre2_match_context* matchContext;

    std::vector<std::string> byteToUnicodeTable;
    uint8_t unicodeToByteTable[65];


    // XXX created by llm
    void generateUnicodeToByte();

    // XXX created by llm
    void generateByteToUnicode() ;

    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second);

    void initMaps(simdjson_result<ondemand::document>& doc);
    void initPreTokeniser(simdjson_result<ondemand::document>& doc);
    void fillAddedTokens(simdjson_result<ondemand::document>& doc);
    void fillVocabTokens(simdjson_result<ondemand::document>& doc);
    void fillMergesTokens(simdjson_result<ondemand::document>& doc);
    std::string createSearchRegex();

  public:

    std::vector<std::string_view> specialSeprateTokens;

    void Open(const std::string& tokeniser_json);

    void Tokenise(const std::string& prompt_str , std::vector<uint32_t>& tokens);
    std::optional<std::string> Decode(uint32_t token_id);

    void Delete();
};
