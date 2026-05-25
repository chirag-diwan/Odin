#include "model_utils.hpp"
#include "ggufreader.hpp"
#include "llama3_tokenizer.hpp"
#include "span.hpp"
#include <string>
#include <sys/mman.h>

int main() {
  GGufReader reader;

  std::string model_path = "/home/chirag/Models/Llama-3.2-1B.Q4_K_M.gguf";
  auto [addr , len]  = reader.OpenFile(model_path);

  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();


  ModelGlobals globals = GetModelGlobals(reader.metadata_key_values);

  LLamaStyleTokenizer tokeniser(globals);

  std::vector<uint32_t> tokens;

  std::string prompt= "Hello how are you";

  tokeniser.Tokenise(prompt, tokens);
  span<uint32_t> token_view(tokens , 0 , tokens.size());
  tokeniser.Decode(token_view);

  munmap(addr, len);

  return 0;
}
