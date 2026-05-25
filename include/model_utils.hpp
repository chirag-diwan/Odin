#pragma once

#include <cstdint>
#include <string_view>
#include "gguf.hpp"
#include "types.hpp"


uint64_t calculateKeyValueCacheByteSize(ModelGlobals& global_struct) {
  auto    head_dimension    = global_struct.embedding_length / global_struct.attention_head_count;
  uint8_t bytes_per_element = 1;
  uint32_t batch_size = 1;

  return global_struct.block_count * batch_size * global_struct.context_length * global_struct.attention_head_count_kv * head_dimension * bytes_per_element;
}


ModelGlobals GetModelGlobals(metadatakv_t& metadata_key_values ){
  ModelGlobals global_struct;
  GGufValue metadata_value ;

  for(const auto& kv : metadata_key_values){
    if(kv.name.find("block_count") != std::string_view::npos){
      global_struct.block_count = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("context_length") != std::string_view::npos){
      global_struct.context_length = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("attention.head_count_kv") != std::string_view::npos){
      global_struct.attention_head_count_kv = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("embedding_length") != std::string_view::npos){
      global_struct.embedding_length = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("attention.head_count") != std::string_view::npos){
      global_struct. attention_head_count = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if( kv.name.find("feed_forward_length") != std::string_view::npos){
      global_struct.feed_forward_length = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if( kv.name.find("rope.freq_base") != std::string_view::npos){
      global_struct.rope_freq_base = Extract<float, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64>(kv.value);

    }else if( kv.name.find("attention.layer_norm_rms_epsilon") != std::string_view::npos){
      global_struct.attention_layer_norm_rms_epsilon = Extract<float, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64>(kv.value);

    }else if(kv.name.find("tokenizer.ggml.eos_token_id") != std::string_view::npos){
      global_struct.ggml_eos_token_id = Extract<uint32_t , GGUF_VALUE_TYPE_INT32 , GGUF_VALUE_TYPE_UINT32>(kv.value);

    }else if(kv.name.find("tokenizer.ggml.bos_token_id") != std::string_view::npos){
      global_struct.ggml_bos_token_id = Extract<uint32_t , GGUF_VALUE_TYPE_INT32 , GGUF_VALUE_TYPE_UINT32>(kv.value);

    }else if(kv.name.find("tokenizer.ggml.tokens") != std::string_view::npos){
      global_struct.token_vocab = &kv.value.array.strings;

    }else if(kv.name.find("tokenizer.ggml.merges") != std::string_view::npos){
      global_struct.token_merges = &kv.value.array.strings;

    }else if(kv.name.find("general.architecture") != std::string_view::npos){
      global_struct.general_model_architecture = kv.value.string;

    }
  }
  return global_struct;
}
