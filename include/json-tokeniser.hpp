#pragma once

#include "logging.hpp"
#include "types.hpp"
#include "unidirectional_map.hpp"
#include "bidirectional_map.hpp"
#include <cstdint>
#include <string>
#include <simdjson.h>
#include <string_view>

using namespace simdjson;

class JsonTokeniser{
  protected:
    TokeniserConfig config;
    PreTokeniser split_tokeniser;


    ondemand::parser parser;
    padded_string json;
    bidirectional_map<std::string_view, uint32_t> vocab;
    bidirectional_map<std::string_view, uint32_t> special_tokens;
    unidirectional_map<uint64_t, merge_rank_result> merges;



    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second){
      return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
    }

  public:
    JsonTokeniser(const std::string& tokeniser_json) :
      json(padded_string::load(tokeniser_json)) 
  {

    auto doc = parser.iterate(json);
    auto added_token = doc["added_tokens"]->get_array();
    auto added_token_count = added_token->count_elements();

    if(added_token_count.has_value()){
      special_tokens = bidirectional_map<std::string_view, uint32_t>(added_token_count.value());
    }else{
      Log(ERROR , "added_token_count dosen't has value");
    }

    for(auto obj : added_token){
      uint32_t id = obj["id"]->get_uint32();
      std::string_view token = obj["content"]->get_string();
      special_tokens.insert(token, id);
    }

    auto pretokenizers = doc["pre_tokenizer"]["pretokenizers"];
    for(auto obj : pretokenizers){
      PreTokeniser pre_tokeniser;
      pre_tokeniser.type = obj["type"]->get_string();
      if(pre_tokeniser.type == "Split"){
        pre_tokeniser.regex = obj["pattern"]["Regex"];
        pre_tokeniser.behavior = obj["behavior"];
        pre_tokeniser.invert = obj["invert"];
        split_tokeniser = pre_tokeniser;
        break;
      }else{
        Log(ERROR , "PreTokeniser type Split not found");
      }
    }

    auto vocab_obj = doc["model"]["vocab"].get_object();
    size_t vocab_size = vocab_obj->count_fields();
    vocab = bidirectional_map<std::string_view, uint32_t>(vocab_size);
    Log(vocab_size);


    auto merges_array = doc["model"]["merges"]->get_array();
    size_t merges_size = merges_array->count_elements();
    merges = unidirectional_map<uint64_t , merge_rank_result>(merges_size);
    Log(merges_size);

    //Re iterate
    auto doc_reinit = parser.iterate(json);


    vocab_obj = doc_reinit["model"]["vocab"].get_object();
    for (auto field : vocab_obj) {
      std::string_view key = field->unescaped_key();
      uint32_t value = uint32_t(field.value());
      vocab.insert(key, value);
    }


    merges_array = doc_reinit["model"]["merges"]->get_array();
    size_t i = 0;
    for(auto element : merges_array){
      std::string_view merge_pair = element.get_string();
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
      merges.insert(key , { .merge_rank = static_cast<uint32_t>(i) , .merge_result = *merge_result });
      i++;
    }
  }
};
