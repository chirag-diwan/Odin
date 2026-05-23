#include "../include/unidirectional_map.hpp"
#include "../include/bidirectional_map.hpp"
#include "../include/ggufreader.hpp"
#include "../include/types.hpp"
#include "../include/model_utils.hpp"


__attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second){
  return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
}

int main(){
  GGufReader reader;

  auto [addr , len]  = reader.OpenFile("/home/chirag/Models/qwen2.5-0.5b-instruct-q4_0.gguf");
  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();

  auto globals = GetModelGlobals(reader.metadata_key_values);

  unidirectional_map<MergeRV> merge(globals.token_merges->size());
  bidirectional_map vocab(globals.token_vocab->size());

  for(size_t i = 0 ; i <globals.token_vocab->size() ; i++){
    if(__builtin_expect(!vocab.insert(globals.token_vocab->at(i), i) , false)){
      Log(ERROR , "cannot insert key value" , globals.token_vocab->at(i), i);
    }
  }

  for(size_t i = 0 ; i <globals.token_merges->size() ; i++){
    std::string_view merge_pair = globals.token_merges->at(i);
    auto split_point = merge_pair.find(' ');
    std::string_view first = merge_pair.substr(0 , split_point);
    std::string_view second = merge_pair.substr(split_point + 1);
    auto first_idx = vocab.getValueOf(first);
    auto second_idx = vocab.getValueOf(second);

    if(__builtin_expect(!first_idx.has_value(),false)){
      Log(ERROR , "value not found for key" , first);
      continue;
    }
    if (__builtin_expect(!second_idx.has_value(),false)) {
      Log(ERROR , "value not found for key" , second);
      continue;
    }



    auto key = getKey(*first_idx, *second_idx);
    std::string result;
    result.reserve(first.size() + second.size());

    result.append(first);
    result.append(second);

    auto merge_result = vocab.getValueOf(result);
    if(__builtin_expect(!merge_result.has_value(),false)){
      Log(ERROR , "value not found for key" , result);
      continue;
    }

    merge.insert(key, { .merge_rank = static_cast<int32_t>(i) , .merge_result = *merge_result });
  } 

  munmap(addr, len);
  return 0;
}
