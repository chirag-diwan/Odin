#include "ggufreader.hpp"
#include "model.hpp"
#include "types.hpp"
#include <sys/mman.h>

int main() {
  GGufReader reader_ctx;

  auto addr_len_pair = reader_ctx.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader_ctx.ParseHeader();
  reader_ctx.ParseAllKeyValues();
  reader_ctx.ParseAllTensors();

  Model model(reader_ctx.metadata_key_values);

  ggml_init_params tensor_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };
  ggml_context* tensor_ctx = ggml_init(tensor_ctx_params);

  ggml_init_params kv_ctx_params = {
    .mem_size =  2048ul * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = false
  };
  ggml_context* kv_ctx = ggml_init(kv_ctx_params);

  std::vector<int> tokens = {
    9707, 11, 1246, 525, 498, 30

  };

  model.PopulateBlocks(reader_ctx.tensors, tensor_ctx);
  model.PopulateKVCache(kv_ctx);
  model.Infer(tokens);

  for(const auto token : tokens){
    std::cout << token << ",";
  }

  ggml_free(kv_ctx);
  ggml_free(tensor_ctx);
  munmap(addr_len_pair.addr, addr_len_pair.len);

  return 0;
}
