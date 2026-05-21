#pragma once

#include <cstdint>
#include "gguf.hpp"
#include "types.hpp"


uint64_t calculateKeyValueCacheByteSize(ModelGlobals& global_struct) {
  auto    head_dimension    = global_struct.embedding_length / global_struct.attention_head_count;
  uint8_t bytes_per_element = 1;
  uint32_t batch_size = 1;

  return global_struct.block_count * batch_size * global_struct.context_length * global_struct.attention_head_count_kv * head_dimension * bytes_per_element;
}


ModelGlobals GetModelGlobals(MetadataKV_t& metadata_key_values ){
  ModelGlobals global_struct;
  GGufValue metadata_value ;

  for(const auto& kv : metadata_key_values){
    if(kv.name == "qwen2.block_count"){
      global_struct.block_count = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name == "qwen2.context_length"){
      global_struct.context_length = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name == "qwen2.attention.head_count_kv"){
      global_struct.attention_head_count_kv = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name == "qwen2.embedding_length"){
      global_struct.embedding_length = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name == "qwen2.attention.head_count"){
      global_struct. attention_head_count = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if( kv.name == "qwen2.feed_forward_length"){
      global_struct.feed_forward_length = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if( kv.name == "qwen2.rope.freq_base"){
      global_struct.rope_freq_base = Extract<float, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64>(kv.value);

    }else if( kv.name == "qwen2.attention.layer_norm_rms_epsilon"){
      global_struct.attention_layer_norm_rms_epsilon = Extract<float, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64>(kv.value);
    }else if(kv.name == "tokenizer.ggml.eos_token_id"){
      global_struct.ggml_eos_token_id = Extract<uint32_t , GGUF_VALUE_TYPE_INT32 , GGUF_VALUE_TYPE_UINT32>(kv.value);
    }else if(kv.name == "tokenizer.ggml.bos_token_id"){
      global_struct.ggml_bos_token_id = Extract<uint32_t , GGUF_VALUE_TYPE_INT32 , GGUF_VALUE_TYPE_UINT32>(kv.value);
    }
  }
  Log(INFO ,"attention_layer_norm_rms_epsilon", global_struct.attention_layer_norm_rms_epsilon);
  Log(INFO ,"embedding_length", global_struct.embedding_length);
  Log(INFO ,"attention_head_count_kv", global_struct.attention_head_count_kv);
  Log(INFO ,"attention_head_count", global_struct.attention_head_count);
  Log(INFO ,"rope_freq_base", global_struct.rope_freq_base);
  Log(INFO ,"block_count", global_struct.block_count);
  Log(INFO ,"context_length", global_struct.context_length);
  Log(INFO ,"feed_forward_length", global_struct.feed_forward_length);
  Log(INFO ,"ggml_eos_token_id", global_struct.ggml_eos_token_id);
  Log(INFO ,"ggml_bos_token_id", global_struct.ggml_bos_token_id);
  return global_struct;
}
