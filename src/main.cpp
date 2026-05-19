#include "debug.hpp"
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

  for(const auto& tensor : reader_ctx.tensors){
    debug_print(tensor);
  }

  //Model model(reader_ctx.metadata_key_values);

  //ggml_init_params tensor_ctx_params = {
  //  .mem_size = 10*1024*1024,
  //  .mem_buffer = NULL,
  //  .no_alloc =true 
  //};
  //ggml_context* tensor_ctx = ggml_init(tensor_ctx_params);


  //ggml_init_params kv_ctx_params = {
  //  .mem_size =  2048ul*1024*1024,
  //  .mem_buffer = NULL,
  //  .no_alloc = false
  //};
  //ggml_context* kv_ctx = ggml_init(kv_ctx_params);

  //std::vector<int> tokens = {39 , 56 , 36};//EFG HIJKLMNOPQRSTUVWXYZ

  //model.PopulateBlocks(reader_ctx.tensors,tensor_ctx);
  //model.PopulateKVCache(kv_ctx);
  //model.Infer(tokens);


  //for (const auto & kv : reader_ctx.metadata_key_values) {
  //  if (kv.name == "tokenizer.ggml.tokens") {
  //    auto token_strings = kv.value.array.strings;
  //    for(auto token : tokens){
  //      std::cout << token_strings[token];
  //    }
  //  }
  //}

  //ggml_free(kv_ctx);
  //ggml_free(tensor_ctx);
  munmap(addr_len_pair.addr, addr_len_pair.len);

  return 0;
}
