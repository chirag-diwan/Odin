#include "ggml-cpu.h"
#include "../include/model.hpp"
#include "../include/model_utils.hpp"
#include "../include/tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "ggml.h"
#include <alloca.h>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <arpa/inet.h>

int main() {
  GGufReader reader;

  auto [addr , len]  = reader.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();


  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_gallocr_t prefill_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_t infer_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  ggml_context* static_ctx = ggml_init(static_ctx_params);

  ModelGlobals globals = GetModelGlobals(reader.metadata_key_values);
  Model model;

  model.SetModelGlobals(globals);
  model.SetBackend(backend);
  model.SetPrefillAllocr(prefill_allocr);
  model.SetInferAllocr(infer_allocr);
  model.PopulateBlocks(static_ctx , reader.tensors);
  model.PopulateKVCache(static_ctx);

  Tokeniser tokeniser(globals);
  bool infer_complete = true;
  std::vector<int32_t> tokens;


  int32_t prev_token;
  while(true){
    if(infer_complete){
      std::string prompt;
      std::cout << "\n $ ";
      std::getline(std::cin , prompt);

      tokens.push_back(151644);
      tokeniser.Tokenise("user\n", tokens); 

      tokeniser.Tokenise(prompt, tokens);

      tokens.push_back(globals.ggml_eos_token_id); 
      tokeniser.Tokenise("\n", tokens);

      tokens.push_back(151644);
      tokeniser.Tokenise("assistant\n", tokens);

      model.Prefill(tokens);
      prev_token = tokens.back();
      tokeniser.Decode(prev_token);

      infer_complete = false;
    }else{
      auto next_token = model.Infer(prev_token);
      tokens.emplace_back(next_token);
      prev_token = next_token;

      if(next_token == globals.ggml_eos_token_id){
        infer_complete = true;
        continue;
      }

      tokeniser.Decode(next_token);
    }
  }

  munmap(addr, len);
  return 0;
}
