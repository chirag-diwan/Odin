#pragma once
#include "errors.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

template <class Tokeniser>
void AttachFile(const char* filepath , Tokeniser& tokeniser, std::vector<uint32_t>& tokens){
  std::ifstream file(filepath, std::ios::binary);
  Errorif(!file, "Cannot open file" , filepath);

  auto file_size = std::filesystem::file_size(filepath);

  std::string buffer;
  buffer.resize(file_size);

  file.read(buffer.data(), file_size);
  file.close();


  tokens.insert(tokens.end() , tokeniser.format_block_1.begin() , tokeniser.format_block_1.end());
  tokeniser.Tokenise(filepath, tokens);
  tokeniser.Tokenise(" File content", tokens);
  tokeniser.Tokenise(buffer, tokens);
  tokens.insert(tokens.end() , tokeniser.format_block_2.begin() , tokeniser.format_block_2.end());
}

inline std::string_view strip(std::string_view str, std::string_view whitespace = " \t\n\r\v\f") {
  const auto start = str.find_first_not_of(whitespace);
  if (start == std::string_view::npos) {
    return "";
  }

  const auto end = str.find_last_not_of(whitespace);
  return str.substr(start, end - start + 1);
}
