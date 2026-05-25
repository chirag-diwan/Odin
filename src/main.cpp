#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../include/model.hpp"
#include "../include/model_utils.hpp"
#include "../include/tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "../include/config.hpp"
#include "../include/span.hpp"
#include "ggml.h"
#include "../include/logging.hpp"
#include <string>
#include <sys/mman.h>

int main(int argc , char **argv) {
  if(argc < 2){
    Log("Usage : \n\t ./odin --model /path/to/model --thread 3 --interactive [true / false] --prompt (if interactive false)\"Hey how are you\"\n");
    return 0;
  }

  Config config = ParseConfig(argc, argv);
  GGufReader reader;

  auto [addr , len]  = reader.OpenFile(config.model_path);

  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();


  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, config.thread_count);

  ggml_gallocr_t prefill_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
  ggml_gallocr_t infer_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  ggml_context* static_ctx = ggml_init(static_ctx_params);

  ModelGlobals globals = GetModelGlobals(reader.metadata_key_values);

  Log(globals.ggml_bos_token_id);
  Log(globals.ggml_eos_token_id);

  Model model;

  model.SetModelGlobals(globals);
  model.SetBackend(backend);
  model.SetPrefillAllocr(prefill_allocr);
  model.SetInferAllocr(infer_allocr);
  model.PopulateBlocks(static_ctx , reader.tensors);
  model.PopulateKVCache(static_ctx);

  model.ReserveDecodeMemory();

  Tokeniser tokeniser(globals);

  bool infer_complete = true;
  std::vector<uint32_t> tokens = { 128000, 2028, 374, 279 };

  model.Prefill(tokens);
  model.Infer(tokens);

  for(const auto token : tokens){
    Log(token);
  }

  munmap(addr, len);

  return 0;
}
