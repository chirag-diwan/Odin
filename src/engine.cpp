#include "../include/engine/engine.hpp"
#include "../include/network/multiclient/network_manager.hpp"
#include "../include/model_utils.hpp"
#include "../include/tokeniser/json_tokeniser.hpp"
#include "../include/gguf/ggufreader.hpp"
#include "../include/config.hpp"
#include "../include/logging.hpp"
#include "../include/types.hpp"
#include "../external/replxx/include/replxx.hxx"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/mman.h>
#include <memory>
#include <iostream>

class MmapGuard {
  void* addr;
  size_t len;
  public:
  MmapGuard(void* addr, size_t len) : addr(addr), len(len) {}
  ~MmapGuard() { if (addr) munmap(addr, len); }
};

struct GgmlDeleter {
  void operator()(ggml_context* ctx) const { if (ctx) ggml_free(ctx); }
};
using UniqueGgmlContext = std::unique_ptr<ggml_context, GgmlDeleter>;

std::string FetchPrompt(bool use_network, NetworkManager& manager , replxx::Replxx& rx) {
  if (use_network) {
    auto prompt = manager.read_prompt();
    if(prompt.has_value()){
      return *prompt;
    }
    return "";
  }

  const char* c_input = rx.input("\n $ ");

  if (c_input == nullptr) {
    std::cerr << "\nBye!\n";
    return "!exit";
  }

  return std::string{c_input};
}

static std::atomic<bool> stop = false;

void sigint_handler(int){
  stop.store(true);
}

int main(int argc, char** argv) {
  std::signal(SIGINT , sigint_handler);
  Log(
      "Usage:\n"
      "\t./odin --model <model_path> --thread <num_threads> --tokeniser-json <tokeniser_json_path>\n"
      "\t[--network-path <path>] [--use-network <0|1>] [--temp <float>] [--top-k <int>]\n\n"
      "Options:\n"
      "\t--model            Path to the model file (required)\n"
      "\t--thread           Number of threads to use (optional)\n"
      "\t--tokeniser-json   Path to tokenizer JSON file (required)\n"
      "\t--network-path     Path or endpoint for network input/output (optional)\n"
      "\t--use-network      Enable/disable network mode (\"false\" = off, \"true\" = on)\n"
     );

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
  ggml_backend_cpu_set_n_threads(backend, config.thread_count);

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  UniqueGgmlContext static_ctx(ggml_init(static_ctx_params));

  auto globals = GetModelGlobals(reader.metadata_key_values);
  auto model = CreateModel(static_ctx.get(), reader);

  Engine engine(model, static_ctx.get(), backend);
  engine.ReservePrefillMemory();
  engine.ReserveDecodeMemory();

  BPETokeniser tokeniser(config.tokeniser_json_path);
  std::vector<uint32_t> tokens;

  NetworkManager manager;
  if (config.use_network) {
    manager.start_listen();
  }


  replxx::Replxx rx;

  rx.history_load("history.txt");
  rx.install_window_change_handler();
  rx.set_max_history_size(1000);

  rx.bind_key(
      replxx::Replxx::KEY::ENTER,
      [&rx](char32_t) {
      rx.invoke(replxx::Replxx::ACTION::INSERT_CHARACTER, '\n');
      return replxx::Replxx::ACTION_RESULT::CONTINUE;
      }
      );

  rx.bind_key(
      replxx::Replxx::KEY::control('S'),
      [](char32_t) {
      return replxx::Replxx::ACTION_RESULT::RETURN;
      }
      );

  while (!stop) {
    std::string prompt = FetchPrompt(config.use_network, manager , rx);
    if (prompt.empty()) {
      continue;
    }
    rx.history_add(prompt);

    if (prompt == "!exit") break;

    size_t last_index = tokens.size();
    tokeniser.Tokenise(prompt, tokens);

    size_t span_size = tokens.size() - last_index;
    span<uint32_t> tokens_view(tokens.data() + last_index, span_size);

    uint32_t next_token = engine.Prefill(tokens_view);
    tokens.push_back(next_token);
    auto tok = tokeniser.Decode(next_token);
    if(tok.has_value())std::cerr << *tok;

    while (next_token != globals.ggml_eos_token_id) {
      next_token = engine.Infer(tokens.back());
      tokens.push_back(next_token);

      if (next_token != globals.ggml_eos_token_id) {
        auto tok = tokeniser.Decode(next_token);
        if(tok.has_value())std::cerr << (*tok);
      }
    }
  }

  rx.history_save("history.txt");
  return EXIT_SUCCESS;
}

