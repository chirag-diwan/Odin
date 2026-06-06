#include "../include/json-tokeniser.hpp"
#include <sys/mman.h>

int main(){
  BPETokeniser tokeniser("/home/chirag/Models/llama3tok.json");
  std::vector<uint32_t> tokens;
  tokeniser.Tokenise("<|begin_of_text|>Hello<|end_of_text|>",tokens);
  for(const auto tok : tokens){
    Log(tok);
  }
}
