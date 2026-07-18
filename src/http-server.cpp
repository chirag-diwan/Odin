#include "../include/engine.hpp"
#include "../include/model_utils.hpp"
#include "../include/json_tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "../include/config.hpp"
#include "../include/logging.hpp"
#include "../include/types.hpp"
#include "../include/welcome.hpp"
#include "../include/formatter.hpp"
#include "../include/http-manager.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "main-utility.hpp"
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/mman.h>

static std::sig_atomic_t interupt = false;

void sig_int_handler(int){
  interupt = true;
}

void abort_callback(const char * message){
  Log(ERROR , "failed with" , message);
}

int main(int argc, char** argv) {
  ggml_set_abort_callback(abort_callback);

  std::signal(SIGINT , sig_int_handler);

  if (argc < 2) {
    return EXIT_FAILURE;
  }

  Config config = ParseConfig(argc, argv);
  GGufReader reader;

  auto [addr, len] = reader.OpenFile(config.model_path);
  MmapGuard mmap_guard(addr, len); 

  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();

  ggml_backend_t backend = ggml_backend_cpu_init();
  auto threadpool_params = ggml_threadpool_params_default(std::thread::hardware_concurrency());

  UniqueThreadpool threadpool(ggml_threadpool_new(&threadpool_params));

  ggml_backend_cpu_set_threadpool(backend, threadpool.get());

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  UniqueGgmlContext static_ctx(ggml_init(static_ctx_params));

  auto globals = GetModelGlobals(reader.metadata_key_values);
  if(globals.general_model_architecture == Architecture::UNKNOWN){
    //TODO Try and get more information about the Architecture using the full name.
    Log(ERROR , "Unknown model architecture" , globals.full_architecture_name);
    return -1;
  }
  auto model = CreateModel(static_ctx.get(), reader);

  Engine engine(model, static_ctx.get(), backend);
  engine.ReservePrefillMemory();
  engine.ReserveDecodeMemory();

  BPETokeniser tokeniser(config.tokeniser_json_path);
  std::vector<uint32_t> tokens;


  PrintHome();

  HttpManager manager(interupt);

  manager.start_listen();

  std::string system_prompt = "You are a helpfull AI agent";

  while (!interupt) {
    std::string raw_prompt = manager.read_prompt();
    if (raw_prompt.empty()) {
      continue;
    }

    if (raw_prompt.starts_with("!exit")) break;
    if(raw_prompt.starts_with("!system")) {
      system_prompt = raw_prompt.substr(7);
    }
    if(raw_prompt.starts_with("!clear-context")){
      engine.ClearContext();
    }

    auto prompt = Formatter::GetFormatted(model.globals.general_model_architecture,system_prompt, raw_prompt);

    size_t last_index = tokens.size();
    tokeniser.Tokenise(prompt, tokens);

    size_t span_size = tokens.size() - last_index;
    std::span<uint32_t> tokens_view(tokens.data() + last_index, span_size);

    uint32_t next_token = engine.Prefill(tokens_view);
    tokens.push_back(next_token);
    auto tok = tokeniser.Decode(next_token);
    if(tok.has_value()){
      manager.write_infered(*tok);
    }

    while (!interupt && (next_token != globals.ggml_eos_token_id)) {
      next_token = engine.Infer(tokens.back());
      tokens.push_back(next_token);

      if (next_token != globals.ggml_eos_token_id) {
        auto tok = tokeniser.Decode(next_token);
        if(tok.has_value()){
          manager.write_infered(*tok);
        }
      }
    }

    manager.write_infered(manager.EOS);
    interupt = false;
  }
  Log("\nBye!\n");

  manager.stop();

  return EXIT_SUCCESS;
}
