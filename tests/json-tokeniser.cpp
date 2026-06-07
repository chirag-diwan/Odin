#include "../include/json_tokeniser.hpp"
#include <sys/mman.h>

int main(){
  BPETokeniser tokeniser("/home/chirag/Models/llama3tok.json");
  for(const auto [tok , id] : tokeniser.special_tokens){
    Log(tok);
  }
}
