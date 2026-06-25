#include "../include/engine/engine.hpp"
#include "../include/ipc/ipc_manager.hpp"
#include "../include/model_utils.hpp"
#include "../include/tokeniser/json_tokeniser.hpp"
#include "../include/gguf/ggufreader.hpp"
#include "../include/config.hpp"
#include "../include/logging.hpp"
#include "../include/types.hpp"
#include "../include/welcome.hpp"
#include "../external/replxx/include/replxx.hxx"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/mman.h>
#include <memory>
#include <iostream>

static std::sig_atomic_t interupt = false;

void sig_int_handler(int){
  interupt = true;
}

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

std::string FetchPrompt(bool use_ipc, IPCManager& manager , replxx::Replxx& rx) {
  if (use_ipc) {
    auto prompt = manager.read_prompt();
    return prompt;
  }

  const char* c_input = rx.input("\n $ ");

  if (c_input == nullptr) {
    return "!exit";
  }

  return std::string{c_input};
}

int main(int argc, char** argv) {
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



  replxx::Replxx rx;

  rx.history_load(config.history_path);
  rx.install_window_change_handler();
  rx.set_max_history_size(1000);

  rx.clear_screen();

  printLogo();
  Log(usage);


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

  IPCManager manager(interupt , config.ipc_path);

  if (config.use_ipc) {
    manager.start_listen();
  }

  while (!interupt) {
    std::string prompt = FetchPrompt(config.use_ipc, manager , rx);
    if (prompt.empty()) {
      continue;
    }

    rx.history_add(prompt);

    if (prompt.starts_with("!exit")) break;

    size_t last_index = tokens.size();
    tokeniser.Tokenise(prompt, tokens);

    size_t span_size = tokens.size() - last_index;
    span<uint32_t> tokens_view(tokens.data() + last_index, span_size);

    uint32_t next_token = engine.Prefill(tokens_view);
    tokens.push_back(next_token);
    auto tok = tokeniser.Decode(next_token);
    if(tok.has_value()){
      if(config.use_ipc){
        manager.write_infered(*tok);
      }
      std::cerr << *tok;
    }

    while (!interupt && (next_token != globals.ggml_eos_token_id)) {
      next_token = engine.Infer(tokens.back());
      tokens.push_back(next_token);

      if (next_token != globals.ggml_eos_token_id) {
        auto tok = tokeniser.Decode(next_token);
        if(tok.has_value()){
          if(config.use_ipc){
            manager.write_infered(*tok);
          }
          std::cerr << *tok;
        }
      }
    }
    interupt = false;
  }
  Log("\nBye!\n");

  manager.stop();

  rx.history_save(config.history_path);

  return EXIT_SUCCESS;
}

