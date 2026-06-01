#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "../include/model.hpp"
#include "../include/model_utils.hpp"
#include "../include/qwen2_tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "../include/config.hpp"
#include "ggml.h"
#include "../include/logging.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <string>
#include <sys/mman.h>


struct Recorder{
  std::chrono::time_point<std::chrono::system_clock> begin;
  std::chrono::time_point<std::chrono::system_clock> end;
  uint64_t time;


  void start(){
    begin = std::chrono::high_resolution_clock::now();
  }

  void stop(){
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
  }
};

int main(int argc , char **argv) {

  Recorder rec;
  Log(INFO , "TIME PER TASK");

  if(argc < 2){
    Log("Usage : \n\t ./odin --model /path/to/model --thread 3 \n");
    return 0;
  }
  Config config = ParseConfig(argc, argv);
  GGufReader reader;

  rec.start();
  auto [addr , len]  = reader.OpenFile(config.model_path);

  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();
  rec.stop();

  Log("Parsing the gguf file" , rec.time, "ms");

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

  rec.start();
  model.SetModelGlobals(globals);
  model.SetBackend(backend);
  model.SetPrefillAllocr(prefill_allocr);
  model.SetInferAllocr(infer_allocr);
  model.PopulateBlocks(static_ctx , reader.tensors);
  model.PopulateKVCache(static_ctx);

  model.ReserveDecodeMemory();

  rec.stop();
  Log("Initializing the model" , rec.time, "ms");

  rec.start();
  QwenStyleTokenizer tokeniser(globals);
  rec.stop();
  Log("Setting up the tokeniser" , rec.time, "ms");


  std::vector<uint32_t> tokens;

  size_t last_index = 0;
  bool infer_complete = true;
  uint64_t tft = 0;
  bool first = true;

  rec.start();
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

      span<uint32_t> tokens_view(tokens.data() + last_index, tokens.size() - last_index);

      if(first){
        rec.start();
      }
      auto next_token = model.Prefill(tokens_view);
      tokens.emplace_back(next_token);

      if(first){
        rec.stop();
        tft = rec.time;
        first = false;
      }

      tokeniser.Decode(next_token);

      infer_complete = false;
    }else{
      auto next_token = model.Infer(tokens.back());
      tokens.emplace_back(next_token);

      if(next_token == globals.ggml_eos_token_id){
        infer_complete = true;
        continue;
      }

      tokeniser.Decode(next_token);
    }
  }
  rec.stop();

  Log("Time to first token" , tft, "ms");
  Log("Tokens per second" , 1000*tokens.size()/rec.time , "tokens/s");


  ggml_gallocr_free(prefill_allocr);
  ggml_gallocr_free(infer_allocr);
  ggml_free(static_ctx);
  munmap(addr, len);

  return 0;
}
