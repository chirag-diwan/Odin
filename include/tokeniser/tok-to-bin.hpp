#pragma once

#include "../logging.hpp"
#include "../types.hpp"
#include "../data_structures/unidirectional_map.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include "../../external/simdjson/simdjson.h"
#include <string_view>
#include <type_traits>

using namespace simdjson;


class Binmaker{
  private:
    std::vector<std::string_view> chunks;
    std::vector<uint32_t> bytes;


  protected:
    TokeniserConfig config;
    PreTokeniser split_tokeniser;

    ondemand::parser parser;
    const padded_string json;

    unidirectional_map<std::string_view, uint32_t> vocab;
    unidirectional_map<std::string_view, uint32_t> special_tokens;
    unidirectional_map<uint64_t, merge_rank_result> merges;

    __attribute__((always_inline)) inline uint64_t getKey(uint32_t first, uint32_t second){
      return (static_cast<uint64_t>(first) << 32) ^ static_cast<uint64_t>(second);
    }

    void init_maps(simdjson_result<ondemand::document>& doc){
      auto added_token = doc["added_tokens"]->get_array();
      size_t added_token_size = added_token->count_elements();
      special_tokens = unidirectional_map<std::string_view, uint32_t>(added_token_size);

      auto vocab_obj = doc["model"]["vocab"].get_object();
      size_t vocab_size = vocab_obj->count_fields();
      vocab = unidirectional_map<std::string_view, uint32_t>(vocab_size);


      auto merges_array = doc["model"]["merges"]->get_array();
      size_t merges_size = merges_array->count_elements();
      merges = unidirectional_map<uint64_t , merge_rank_result>(merges_size);

    }

    void init_pre_tokeniser(simdjson_result<ondemand::document>& doc){
      auto pretokenizers = doc["pre_tokenizer"]["pretokenizers"];
      for(auto obj : pretokenizers){
        std::string_view type = obj["type"]->get_string();
        if(type == "Split"){
          split_tokeniser.regex = obj["pattern"]["Regex"]->get_string();
          split_tokeniser.behavior = obj["behavior"]->get_string();
          split_tokeniser.invert = obj["invert"]->get_bool();
          break;
        }else{
          Log(ERROR , "PreTokeniser type Split not found");
        }
      }
    }

    void fill_added_tokens(simdjson_result<ondemand::document>& doc){
      auto added_token = doc["added_tokens"]->get_array();
      for(auto obj : added_token){
        uint32_t id = obj["id"]->get_uint32();
        std::string_view token = obj["content"]->get_string();
        special_tokens.insert(token, id);
      }
    }


    void fill_vocab_tokens(simdjson_result<ondemand::document>& doc){
      auto vocab_obj = doc["model"]["vocab"].get_object();
      for (auto field : vocab_obj) {
        std::string_view key = field->unescaped_key();
        uint32_t value = uint32_t(field.value());
        vocab.insert(key, value);
      }
    }


    void fill_merges_tokens(simdjson_result<ondemand::document>& doc){
      auto merges_array = doc["model"]["merges"]->get_array();
      uint32_t i = 0;
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

        merges.insert(key , { .merge_rank = i , .merge_result = *merge_result });
        if(i == std::numeric_limits<uint32_t>::max()){
          Log(WARN,"merge rank overflowed uint32_t");
          break;
        }
        i++;
      }
    }


  public:
    Binmaker(const std::string& tokeniser_json) : json(padded_string::load(tokeniser_json)) {
      auto doc = parser.iterate(json);
      init_maps(doc);

      //Re iterate
      auto doc_reinit = parser.iterate(json);

      fill_added_tokens(doc_reinit);
      init_pre_tokeniser(doc_reinit);
      fill_vocab_tokens(doc_reinit);
      fill_merges_tokens(doc_reinit);
    }

    ~Binmaker(){
    }
};

