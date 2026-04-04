#include "GGUF/gguf.h"
#include <ggml-cpu.h>
#include <ggml.h>

int main() {

  Odin::GGufReader ctx;
  ctx.openFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  ctx.parseHeader();
  ctx.parseAllKeyValues();

  struct ggml_init_params weight_params = {.mem_size = sizeof(ggml_tensor) *
                                                       ctx.header.tensor_count,
                                           .mem_buffer = NULL,
                                           .no_alloc   = false};
  struct ggml_context*    weight_ctx    = ggml_init(weight_params);

  ctx.parseAllTensors(weight_ctx);
}
