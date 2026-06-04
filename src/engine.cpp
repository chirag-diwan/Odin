#include "../include/engine.hpp"
#include "../include/model_utils.hpp"
#include "../include/qwen2_tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "../include/config.hpp"
#include "../include/logging.hpp"
#include "../include/types.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cstdint>
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

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  ggml_context* static_ctx = ggml_init(static_ctx_params);

  auto globals = GetModelGlobals(reader.metadata_key_values);
  auto model_ = CreateModel(static_ctx, reader);

  Engine engine(model_ , static_ctx , backend);
  engine.ReservePrefillMemory();
  engine.ReserveDecodeMemory();

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

      last_index = tokens.size();

      tokeniser.TokeniseFormatted(prompt, tokens);

      const size_t span_size = tokens.size() - last_index;

      span<uint32_t> tokens_view(tokens.data() + last_index , span_size);
      auto next_token = engine.Prefill(tokens_view);
      tokens.emplace_back(next_token);


      tokeniser.Decode(next_token);

      infer_complete = false;
    }else{
      auto next_token = engine.Infer(tokens.back());
      tokens.emplace_back(next_token);

      if(next_token == globals.ggml_eos_token_id){
        infer_complete = true;
        continue;
      }

      tokeniser.Decode(next_token);
    }
  }


  ggml_free(static_ctx);
  munmap(addr, len);

  return 0;
}
