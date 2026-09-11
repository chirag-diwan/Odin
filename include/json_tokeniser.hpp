#pragma once

#include "./types.hpp"
#include "./data_structures/UnidirectionalMap.hpp"
#include "./data_structures/BidirectionalMap.hpp"
#include "./data_structures/DoublyLinkedList.hpp"
#include "./data_structures/Automaton.hpp"
#include "data_structures/ConstexprString.hpp"
#include <cstdint>
#include <optional>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <string>
#include "simdjson/simdjson.h"
#include <string_view>

using namespace simdjson;

pcre2_code* compile_regex(const std::string_view& regex);

consteval std::array<ConstexprStr<3> , 256> generateByteToUnicode() {
  std::array<ConstexprStr<3> , 256> byteToUnicodeTable{};
  int n = 0;
  for (int b = 0; b < 256; b++) {
    ConstexprStr<3> str{};
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
      str[0] = b;
      str.set_size(1);
      byteToUnicodeTable[b] = str;
    } else {
      int unicode_val = 256 + n;
      ConstexprStr<3> utf8_char{};
      utf8_char[0] = (static_cast<char>(0xC0 | (unicode_val >> 6)));
      utf8_char[1] = (static_cast<char>(0x80 | (unicode_val & 0x3F)));
      utf8_char.set_size(2);

      byteToUnicodeTable[b] = utf8_char;
      n++;
    }
  }
  return byteToUnicodeTable;
}

consteval std::array<uint8_t , 68> generateUnicodeToByte(){
  std::array<uint8_t , 68> unicodeToByteTable{};
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
    } else {
      int unicode_val = 256 + n;
      unicodeToByteTable[unicode_val - 256] = static_cast<uint8_t>(b);
      n++;
    }
  }
  return unicodeToByteTable;
}

class BPETokeniser{
  private:
    std::vector<std::string_view> chunks;
    std::vector<uint32_t> bytes {};
    DoublyLinkedList<uint32_t> bytes_dll {};

    TokeniserConfig config;
    PreTokeniser split_tokeniser;

    //ondemand::parser parser;
    padded_string json;
    dom::parser parserDOM;

    void* file;
    size_t size;

    BidirectionalMap<std::string_view, uint32_t> specialTokens;
    UnidirectionalMap<uint64_t, merge_rank_result> merges;


    pcre2_code* preTokRegex;
    Automaton automaton;

    pcre2_jit_stack* jitStack;
    pcre2_match_context* matchContext;

    pcre2_match_data* preTokMatchData;
    pcre2_match_data* specialTokMatchData;

    inline static constexpr auto byteToUnicodeTable = generateByteToUnicode();
    inline static constexpr auto unicodeToByteTable = generateUnicodeToByte();


    ODIN_INLINE uint64_t getKey(uint32_t first, uint32_t second);

    void initMaps(simdjson_result<dom::element>& doc);
    void initPreTokeniser(simdjson_result<dom::element>& doc);
    void fillAddedTokens(simdjson_result<dom::element>& doc);
    void fillVocabTokens(simdjson_result<dom::element>& doc);
    void fillMergesTokens(simdjson_result<dom::element>& doc);
    std::string createSearchRegex();

    void tokeniseVec(const std::string& prompt_str , std::vector<uint32_t>& tokens);

  public:
    BidirectionalMap<std::string_view, uint32_t> vocab;
    std::vector<std::string_view> specialSeprateTokens;

    void Open(const std::string& tokeniser_json);
    void OpenDOM(const std::string& tokeniser_json);

    void TokeniseDLL(const std::string& prompt_str , std::vector<uint32_t>& tokens);
    std::optional<std::string> Decode(uint32_t token_id);

    void Delete();
};
