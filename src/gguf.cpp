#include "../include/ggufreader.hpp"
#include "../include/logging.hpp"
#include "../include/debug.hpp"
#include <string>
#include <sys/mman.h>

int main(int argc , char **argv) {
  if(argc < 2){
    Log("Usage : \n\t ./odin --model /path/to/model --thread 3 --interactive [true / false] --prompt (if interactive false)\"Hey how are you\"\n");
    return 0;
  }
  GGufReader reader;

  auto [addr , len]  = reader.OpenFile(argv[2]);

  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();


  for(const auto& kv : reader.metadata_key_values){
    Log(INFO , "Key Name :", kv.name);
    debug_print(kv.value);
  }

  for(const auto& tensor : reader.tensors){
    debug_print(tensor);
  }

  munmap(addr, len);

  return 0;
}
