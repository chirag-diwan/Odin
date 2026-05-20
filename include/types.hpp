#pragma once
#include "ggml.h"
#include <cstdint>
#include <string_view>
#include <vector>


#define DIM_ARRAY_MAX_SIZE 8

struct AddrLenPair{
  void * addr;
  int64_t len;

  AddrLenPair(){
    addr = nullptr;
    len = 0;
  }

  AddrLenPair(void * addr , int64_t len) : addr(addr) , len(len){}

};

struct GGufHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;

  GGufHeader(){
    magic = 0;
    version = 0;
    tensor_count = 0;
    metadata_kv_count = 0;
  }

};



enum GGufValueType {
  GGUF_VALUE_TYPE_UINT8   = 0,
  GGUF_VALUE_TYPE_INT8    = 1,
  GGUF_VALUE_TYPE_UINT16  = 2,
  GGUF_VALUE_TYPE_INT16   = 3,
  GGUF_VALUE_TYPE_UINT32  = 4,
  GGUF_VALUE_TYPE_INT32   = 5,
  GGUF_VALUE_TYPE_FLOAT32 = 6,
  GGUF_VALUE_TYPE_BOOL    = 7,
  GGUF_VALUE_TYPE_STRING  = 8,
  GGUF_VALUE_TYPE_ARRAY   = 9,
  GGUF_VALUE_TYPE_UINT64  = 10,
  GGUF_VALUE_TYPE_INT64   = 11,
  GGUF_VALUE_TYPE_FLOAT64 = 12,
  GGUF_VALUE_TYPE_NULL,
};

struct GGufArray{
  uint32_t elem_type;
  uint8_t* data;
  uint64_t length;
  std::vector<std::string_view> strings;

  GGufArray(){
    elem_type = GGUF_VALUE_TYPE_NULL;
    data = nullptr;
    length = 0;
    strings = {};
  }
};

struct GGufValue {
  uint8_t* data;
  std::string_view string;
  GGufArray array;
  uint32_t type;

  GGufValue(){
    data = nullptr;
    type = GGUF_VALUE_TYPE_NULL;
  }
};


struct metadata_keyvalue{
  std::string_view name;
  GGufValue value;
};

using MetadataKV_t = std::vector<metadata_keyvalue>;


struct GGufTensor {
  std::string_view name;
  ggml_type        tensor_type;
  uint32_t         dimension_count;
  int64_t          dimensions[DIM_ARRAY_MAX_SIZE];
  uint64_t         file_offset;
  uint64_t         byte_size;
  uint8_t*         weights_data;

  GGufTensor(){
    dimension_count = 1;
    for(int i = 0 ; i < DIM_ARRAY_MAX_SIZE ; i++){
      dimensions[i] = 1;
    }
    file_offset = 0;
    byte_size = 1;
    weights_data = 0;
  }
};


struct ModelGlobals{
  uint32_t block_count;
  uint32_t embedding_length;
  uint32_t feed_forward_length;
  uint32_t attention_head_count;
  uint32_t attention_head_count_kv;
  uint32_t context_length ;
  double rope_freq_base ;
  double attention_layer_norm_rms_epsilon ;

  ModelGlobals(){
    block_count = 0;
    embedding_length = 0;
    feed_forward_length = 0;
    attention_head_count = 0;
    attention_head_count_kv = 0;
    context_length  = 0;
    rope_freq_base  = 0;
    attention_layer_norm_rms_epsilon  = 0;
  }
};

struct ModelGlobalTensors {
  struct ggml_tensor* token_embd_weights  ;
  struct ggml_tensor* output_norm_weights ;
  struct ggml_tensor* output_weights      ;

  ModelGlobalTensors(){
    token_embd_weights  = nullptr;
    output_norm_weights = nullptr;
    output_weights      = nullptr;
  }
};


struct ModelState{
  bool isPreFilled;
  ModelState(){
    isPreFilled = false;
  }
};
