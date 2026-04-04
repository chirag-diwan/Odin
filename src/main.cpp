#include "GGUF/gguf.h"
#include <ggml-cpu.h>
#include <ggml.h>

int main() {

  Odin::GGUF ctx;
  ctx.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  ctx.ParseHeader();
  ctx.ParseKeyValue();

  struct ggml_init_params weight_params = {.mem_size = sizeof(ggml_tensor) *
                                                       ctx.header.tensor_count,
                                           .mem_buffer = NULL,
                                           .no_alloc   = false};
  struct ggml_context*    weight_ctx    = ggml_init(weight_params);

  ctx.ParseTensors(weight_ctx);
}
