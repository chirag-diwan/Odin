#include "../include/ggufreader.hpp"
#include "../include/tokeniser.hpp"
#include "../include/model_utils.hpp"
#include <sys/mman.h>

int main(){
  GGufReader reader;
  std::string model_path = "/home/chirag/Models/Llama-3.2-1B.Q4_K_M.gguf";
  auto [addr , len] = reader.OpenFile(model_path);
  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();

  ModelGlobals globals = GetModelGlobals(reader.metadata_key_values);

  Tokeniser t(globals);
  std::vector<uint32_t> tokens;
  t.Tokenise("Hello how are you my friend ?", tokens);
  for(const auto token : tokens){
    std::cout << token << ',';
  }

  munmap(addr, len);
}
