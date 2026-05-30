#include "../include/model_utils.hpp"
#include "../include/qwen2_tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "../include/vector.hpp"
#include <string>
#include <sys/mman.h>

int main(int argc , char **argv) {
  GGufReader reader;

  std::string model_path = "/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf";
  auto [addr , len]  = reader.OpenFile(model_path);

  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();



  ModelGlobals globals = GetModelGlobals(reader.metadata_key_values);

  QwenStyleTokenizer tokeniser(globals);

  vector<uint32_t> tokens;

  std::string prompt = "Hello world";
  tokeniser.TokeniseFormat(prompt, tokens);

  munmap(addr, len);

  return 0;
}
