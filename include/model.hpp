#pragma once

#include <cstdint>
#include "./types.hpp"
#include "gguf.hpp"

struct ModelGlobals{
  uint32_t block_count ;
  uint32_t sequence_length ;
  uint32_t head_count_kv ;
  uint32_t embedding_length ;
  uint32_t attention_head_count ;
};


uint64_t calculateKeyValueCacheByteSize(ModelGlobals& global_struct) {
  auto    head_dimension    = global_struct.embedding_length / global_struct.attention_head_count;
  uint8_t bytes_per_element = 1;
  uint32_t batch_size = 1;

  return 2 * global_struct.block_count * batch_size * global_struct.sequence_length * global_struct.head_count_kv * head_dimension * bytes_per_element;
}

struct ModelGlobalTensors {
  struct ggml_tensor* token_embd_weights  = nullptr;
  struct ggml_tensor* output_norm_weights = nullptr;
  struct ggml_tensor* output_weights      = nullptr;
};


void SetModelGlobals(MetadataKV_t& metadata_key_values , ModelGlobals& global_struct){
  GGufValue metadata_value ;

  metadata_value = metadata_key_values.at("qwen2.block_count");
  global_struct.block_count =
    Odin::Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(metadata_value);

  metadata_value = metadata_key_values.at("qwen2.context_length");
  global_struct.sequence_length =
    Odin::Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(metadata_value);

  metadata_value = metadata_key_values.at("qwen2.attention.head_count_kv");
  global_struct. head_count_kv =
    Odin::Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(metadata_value);

  metadata_value = metadata_key_values.at("qwen2.embedding_length");
  global_struct. embedding_length =
    Odin::Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(metadata_value);

  metadata_value = metadata_key_values.at("qwen2.attention.head_count");
  global_struct. attention_head_count =
    Odin::Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(metadata_value);

}
