#include "../include/bidirectional_map.hpp"
#include "../include/ggufreader.hpp"
int main(){
  GGufReader reader;

  auto [addr , len]  = reader.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();

  utils::bidirectional_map map(reader.metadata_key_values.size());
  for(const auto& kv : reader.metadata_key_values){
    if(kv.name == "tokenizer.ggml.tokens"){
      for(size_t i = 0 ; i <kv.value.array.strings.size() ; i++){
        map.insert(kv.value.array.strings[i], i);
      }
    }
  }
  munmap(addr, len);
  return 0;
}
