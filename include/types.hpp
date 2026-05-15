#pragma once
#include "ggml.h"
#include <cstdint>
#include <string_view>
#include <unordered_map>


#define DIM_ARRAY_MAX_SIZE 8

struct AddrLenPair{
  void * addr;
  int64_t len;
};

struct GGufHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
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

struct GGufString {
  uint64_t length;
  uint8_t * data;

  GGufString(){
    length = 0;
    data = nullptr;
  }
};

struct GGufValue {
  uint8_t* data;
  GGufString string;
  GGufString array;
  uint32_t type;

  GGufValue(){
    data = nullptr;
    type = GGUF_VALUE_TYPE_NULL;
  }
};

using MetadataKV_t = std::unordered_map<std::string_view, GGufValue>;


struct GGufTensor {
  std::string_view name;
  ggml_type        tensor_type;
  uint32_t         dimension_count;
  int64_t          dimensions[DIM_ARRAY_MAX_SIZE];
  uint64_t         file_offset;
  uint64_t         byte_size;
  uint8_t*         weights_data;
};
