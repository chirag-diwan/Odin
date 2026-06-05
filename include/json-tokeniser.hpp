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
    ondemand::document doc;
    bidirectional_map<std::string_view, uint32_t> vocab;
    bidirectional_map<std::string_view, uint32_t> special_tokens;
    unidirectional_map<uint32_t, merge_rank_result> merges;



    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second){
      return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
    }

  public:
    JsonTokeniser(const std::string& tokeniser_json) :
      json(padded_string::load(tokeniser_json)) ,
      doc(parser.iterate(json))
  {
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
    auto vocab_field_count = vocab_obj->count_fields();
    if(vocab_field_count.has_value()){
      vocab = bidirectional_map<std::string_view, uint32_t>(vocab_field_count.value());
    }else{
      Log(ERROR , "vocab_field_count dosen't has value");
    }

    for (auto field : vocab_obj) {
      std::string_view key = field.unescaped_key(); // or field.key()
      uint32_t value = uint32_t(field.value());
      vocab.insert(key, value);
    }

    auto merges_array = doc["model"]["merges"]->get_array();

    std::vector<std::pair<std::string_view, std::string_view>> temp;
    temp.reserve(1024); // optional guess

    for (auto element : merges_array) {
      std::string_view merge_pair = element.get_string().value();

      auto split_point = merge_pair.find(' ');
      if (split_point == std::string_view::npos) continue;

      temp.emplace_back(
          merge_pair.substr(0, split_point),
          merge_pair.substr(split_point + 1)
          );
    }

    merges = unidirectional_map<uint32_t , merge_rank_result>(temp.size());

    for(size_t i = 0 ; i < temp.size() ; i++){
      std::string_view first = temp[i].first;
      std::string_view second = temp[i].second;
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
    }
  }
};
