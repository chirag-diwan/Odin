#include "../include/json_tokeniser.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main() {
  const std::string tokenizer_path =
    "/home/chirag/Models/llama3tok.json";

  BPETokeniser tokeniser(tokenizer_path);

  // Read the tokenizer file into memory
  auto test_path = "/home/chirag/Models/train.jsonl";
  std::ifstream file(test_path, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open " << test_path << '\n';
    return 1;
  }

  std::string contents(
      (std::istreambuf_iterator<char>(file)),
      std::istreambuf_iterator<char>());

  std::vector<uint32_t> tokens;

  auto start = std::chrono::high_resolution_clock::now();

  tokeniser.Tokenise(contents, tokens);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed = end - start;

  double seconds = elapsed.count();
  double mb = contents.size() / (1024.0 * 1024.0);

  std::cout << "Input size: " << contents.size() << " bytes\n";
  std::cout << "Tokens: " << tokens.size() << '\n';
  std::cout << "Time: " << seconds * 1000.0 << " ms\n";
  std::cout << "Throughput: " << mb / seconds << " MB/s\n";
  std::cout << "Token rate: " << tokens.size() / seconds
    << " tokens/s\n";
}
