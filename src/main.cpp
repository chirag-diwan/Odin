#include "ggml.h"
#include "ggufreader.hpp"
#include "model.hpp"
#include "types.hpp"

int main() {
  GGufReader reader_ctx;

  auto addr_len_pair = reader_ctx.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader_ctx.ParseHeader();
  reader_ctx.ParseAllKeyValues();

  reader_ctx.ParseAllTensors();

  Model model(reader_ctx.metadata_key_values );

  ggml_init_params weight_ctx_params = {
    .mem_size = 4096ul*1024*1024,
    .mem_buffer = NULL,
    .no_alloc = false
  };
  ggml_context* weight_ctx = ggml_init(weight_ctx_params);


  ggml_init_params compute_ctx_params = {
    .mem_size = 256*1024*1024,
    .mem_buffer = NULL,
    .no_alloc = false
  };
  ggml_context* compute_ctx = ggml_init(compute_ctx_params);

  std::vector<int> tokens = {36 , 37};

  model.PopulateBlocks(reader_ctx.tensors,weight_ctx);
  model.PopulateKVCache(weight_ctx);
  model.PopulateCausalMask(weight_ctx);
  model.Prefill(compute_ctx , tokens);
  model.Infer(tokens);

  
  for (const auto & kv : reader_ctx.metadata_key_values) {
    if (kv.name == "tokenizer.ggml.tokens") {
      auto token_strings = kv.value.array.strings;
      for(auto token : tokens){
        std::cout << token_strings[token];
      }
    }
  }

  ggml_free(weight_ctx);
  ggml_free(compute_ctx);
  munmap(addr_len_pair.addr, addr_len_pair.len);

  return 0;
}
