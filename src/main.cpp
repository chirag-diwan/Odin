#include "gguf/gguf.h"

int main() {
  GGUF ctx;
  ctx.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  ctx.ParseHeader();
  ctx.ParseKeyValue();

  ctx.ParseTensors();
}
