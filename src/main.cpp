#include "GGUF/gguf.h"
#include "Model/model.h"
#include <ggml-cpu.h>
#include <ggml.h>

using namespace Odin;

int main() {

  GGufReader reader_ctx;
  reader_ctx.openFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  reader_ctx.parseHeader();
  reader_ctx.parseAllKeyValues();

  struct ggml_init_params weight_params = {
      .mem_size   = ggml_tensor_overhead() * reader_ctx.header.tensor_count,
      .mem_buffer = NULL,
      .no_alloc   = true,

  };
  struct ggml_context* weight_ctx = ggml_init(weight_params);

  std::vector<ModelBlock> blocks         = {};
  ModelGlobalTensors      global_tensors = {};

  reader_ctx.parseAllTensors(blocks, global_tensors, weight_ctx);

  Model model(reader_ctx, blocks, global_tensors);
  model.initializeComputeAndCache();

  struct ggml_init_params compute_params = {
      .mem_size   = 512 * 1024 * 1024,
      .mem_buffer = NULL,
      .no_alloc   = false,

  };
  struct ggml_context* compute_context = ggml_init(compute_params);

  std::vector<int32_t> Tokens = {112, 1231};
  model.Start(compute_context, Tokens);
  return 0;
}
