#include "../include/engine/engine.hpp"
#include "../include/network/network_manager.hpp"
#include "../include/model_utils.hpp"
#include "../include/tokeniser/json_tokeniser.hpp"
#include "../include/gguf/ggufreader.hpp"
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

std::string FetchPrompt(bool use_network, NetworkManager& manager) {
  if (use_network) {
    return *manager.read_input();
  }

  std::cout << "\n $ ";
  std::string prompt, line;
  while (std::getline(std::cin, line)) {
    if (line.find("!exit") == 0) return "!exit";
    if (line.find("!submit") == 0) break;
    prompt += line + "\n";
  }
  return prompt;
}

int main(int argc, char** argv) {
  Log(
      "Usage:\n"
      "\t./odin --model <model_path> --thread <num_threads> --tokeniser-json <tokeniser_json_path>\n"
      "\t[--network-path <path>] [--use-network <0|1>] [--temp <float>] [--top-k <int>]\n\n"
      "Options:\n"
      "\t--model            Path to the model file (required)\n"
      "\t--thread           Number of threads to use (required)\n"
      "\t--tokeniser-json   Path to tokenizer JSON file (required)\n"
      "\t--network-path     Path or endpoint for network input/output (optional)\n"
      "\t--use-network      Enable/disable network mode (0 = off, 1 = on)\n"
     );

  if (argc < 2) {
    return EXIT_FAILURE;
  }


  Config config = ParseConfig(argc, argv);
  GGufReader reader;

  auto [addr, len] = reader.OpenFile(config.model_path);
  MmapGuard mmap_guard(addr, len); // Replaces manual munmap

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

  NetworkManager manager(config.network_path.c_str());
  if (config.use_network) {
    manager.start_listen();
  }

  while (true) {
    std::string prompt = FetchPrompt(config.use_network, manager);
    if (prompt == "!exit") break;

    size_t last_index = tokens.size();
    tokeniser.Tokenise(prompt, tokens);

    size_t span_size = tokens.size() - last_index;
    span<uint32_t> tokens_view(tokens.data() + last_index, span_size);

    uint32_t next_token = engine.Prefill(tokens_view);
    tokens.push_back(next_token);
    tokeniser.Decode(next_token);

    while (next_token != globals.ggml_eos_token_id) {
      next_token = engine.Infer(tokens.back());
      tokens.push_back(next_token);

      if (next_token != globals.ggml_eos_token_id) {
        tokeniser.Decode(next_token);
      }
    }
  }

  return EXIT_SUCCESS;
}
