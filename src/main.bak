#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggufreader.hpp"
#include "model.hpp"
#include "model_utils.hpp"
#include "types.hpp"
#include <alloca.h>
#include <cstddef>
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

  std::vector<int> tokens = {
    9707, 11, 1246, 525, 498, 30
  };

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  ggml_context* static_ctx = ggml_init(static_ctx_params);

  ModelGlobals globals = GetModelGlobals(reader.metadata_key_values);
  Model model(globals);

  model.SetBackend(backend);
  model.SetGAlloc(allocr);
  model.PopulateBlocks(static_ctx , reader.tensors);
  model.PopulateKVCache(static_ctx);
  model.Prefill(tokens);
  model.Infer(tokens , 0.9);

  for(const auto token : tokens){
    std::cout << token << ",";
  }

  ggml_free(static_ctx);
  ggml_gallocr_free(allocr);
  ggml_backend_free(backend);
  munmap(addr, len);

  return 0;
}
