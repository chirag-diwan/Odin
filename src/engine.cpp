#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "../include/model.hpp"
#include "../include/model_utils.hpp"
#include "../include/qwen2_tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "../include/config.hpp"
#include "ggml.h"
#include "../include/logging.hpp"
#include <cstdlib>
#include <string>
#include <sys/mman.h>

int main(int argc , char **argv) {
  if(argc < 2){
    Log("Usage : \n\t ./odin --model /path/to/model --thread 3 \n");
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
  Model model;

  model.SetModelGlobals(globals);
  model.SetBackend(backend);
  model.SetPrefillAllocr(prefill_allocr);
  model.SetInferAllocr(infer_allocr);
  model.PopulateBlocks(static_ctx , reader.tensors);
  model.PopulateKVCache(static_ctx);

  model.ReserveDecodeMemory();

  QwenStyleTokenizer tokeniser(globals);


  std::vector<uint32_t> tokens;

  size_t last_index = 0;
  bool infer_complete = true;

  while(true){
    if (infer_complete) {
      std::string prompt;
      std::cout << "\n $ ";
      std::getline(std::cin, prompt);

      if (prompt == "exit") {
        break;
      }

      // 1. Capture the EXACT boundary BEFORE appending new text
      last_index = tokens.size();

      // 2. Append the new prompt tokens to the history
      tokeniser.TokeniseFormatted(prompt, tokens);

      // 3. Slice ONLY the new prompt tokens for the Prefill phase
      span<uint32_t> tokens_view(tokens.data() + last_index, tokens.size() - last_index);

      // 4. Execute prefill on the isolated new context
      auto next_token = model.Prefill(tokens_view);
      tokens.emplace_back(next_token);

      tokeniser.Decode(next_token);

      infer_complete = false;
    }else{  auto next_token = model.Infer(tokens.back());
      tokens.emplace_back(next_token);

      if(next_token == globals.ggml_eos_token_id){
        infer_complete = true;
        continue;
      }

      tokeniser.Decode(next_token);
    }
  }


  ggml_gallocr_free(prefill_allocr);
  ggml_gallocr_free(infer_allocr);
  ggml_free(static_ctx);
  munmap(addr, len);

  return 0;
}
