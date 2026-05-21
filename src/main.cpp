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


  Tokeniser tokeniser(reader.metadata_key_values);
  std::vector<int32_t> user_text_tokens;
  tokeniser.Tokenise("Hello, how are you?", user_text_tokens);

  std::vector<int32_t> final_prompt;

  final_prompt.push_back(151644);
  tokeniser.Tokenise("user\n", final_prompt); 

  final_prompt.insert(final_prompt.end(), user_text_tokens.begin(), user_text_tokens.end());

  final_prompt.push_back(globals.ggml_eos_token_id); 
  tokeniser.Tokenise("\n", final_prompt);

  final_prompt.push_back(151644);
  tokeniser.Tokenise("assistant\n", final_prompt);

  model.Prefill(final_prompt);
  model.Infer(final_prompt);
  tokeniser.Decode(final_prompt);

  munmap(addr, len);
  return 0;
}
