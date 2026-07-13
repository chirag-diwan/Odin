#include <chrono>
#include <iostream>
#include <vector>
#include "../include/tokeniser/json_tokeniser.hpp"

int main() {
  const std::string tokenizer_path =
    "/home/chirag/Models/llama3tok.json";

  BPETokeniser tokeniser(tokenizer_path);

  std::string contents = "This has to be tokenised";

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
