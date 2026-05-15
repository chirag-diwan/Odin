#include "ggufreader.hpp"

int main() {
  Odin::GGufReader reader_ctx;

  auto addr_len_pair = reader_ctx.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader_ctx.ParseHeader();
  reader_ctx.ParseAllKeyValues();
  reader_ctx.ParseAllTensors();

  munmap(addr_len_pair.addr, addr_len_pair.len);

  return 0;
}
