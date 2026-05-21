#include "ggml-cpu.h"
#include "model.hpp"
#include "model_utils.hpp"
#include "tokeniser.hpp"
#include "ggufreader.hpp"
#include <alloca.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <arpa/inet.h>

int main() {
  GGufReader reader;

  auto [addr , len]  = reader.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();

  std::vector<int32_t> tokens;


  Tokeniser t(reader.metadata_key_values);

  t.Tokenise("You are a really help full assitant and you are supposed to answer the queries asked ot you" , tokens);


  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

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
  model.SetGAlloc(allocr);
  model.PopulateBlocks(static_ctx , reader.tensors);
  model.PopulateKVCache(static_ctx);
  model.Prefill(tokens , 0.9);

  std::cout << " $ ";
  std::string new_prompt;
  std::cin >> new_prompt;
  std::vector<int32_t> new_tokens;
  t.Tokenise(new_prompt, new_tokens);
  model.Infer(new_tokens , 0.9);
  t.Decode(new_tokens);

  t.Decode(tokens);

  munmap(addr, len);
  return 0;
}
