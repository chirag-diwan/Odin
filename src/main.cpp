#include "ggml.h"
#include "ggufreader.hpp"
#include "model.hpp"

int main() {
  GGufReader reader_ctx;

  auto addr_len_pair = reader_ctx.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader_ctx.ParseHeader();
  reader_ctx.ParseAllKeyValues();
  reader_ctx.ParseAllTensors();

  Model model(reader_ctx.metadata_key_values);
  std::vector<uint16_t> tokens = {1,2,3};

  ggml_init_params weight_ctx_params = {
    .mem_size = 2048ul*1024*1024,
    .mem_buffer = NULL,
    .no_alloc = false
  };
  ggml_context* weight_ctx = ggml_init(weight_ctx_params);


  ggml_init_params compute_ctx_params = {
    .mem_size = 16*1024*1024,
    .mem_buffer = NULL,
    .no_alloc = false
  };
  ggml_context* compute_ctx = ggml_init(compute_ctx_params);

  model.PopulateBlocks(reader_ctx.tensors,weight_ctx);
  model.PopulateKVCache(weight_ctx);
  model.Prefill(tokens,compute_ctx);
  model.Infer(compute_ctx);


  ggml_free(weight_ctx);
  ggml_free(compute_ctx);
  munmap(addr_len_pair.addr, addr_len_pair.len);
  
  return 0;
}
