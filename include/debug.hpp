#pragma once


#include "ggml.h"
#include "gguf.hpp"
#include "logging.hpp"
#include "types.hpp"

void debug_print(ggml_tensor* t , const char* name){
  Log(name , t->ne[0] , t->ne[1] , t->ne[2] , t->ne[3]);
}

void debug_print(const GGufValue& val){
  if(val.type == GGUF_VALUE_TYPE_ARRAY){
    std::cout << '[';
    auto array_len = val.array.length > 100 ? 100 : val.array.length;
    if(val.array.elem_type == GGUF_VALUE_TYPE_STRING){
      for(size_t i = 0 ; i <array_len  ; i++){
        std::cout << val.array.strings[i];
        if(i != array_len - 1) std::cout << ',';
      }
    } else if(val.type == GGUF_VALUE_TYPE_FLOAT32 || val.type == GGUF_VALUE_TYPE_FLOAT64){
      auto element_size = GGufValueSize(val.array.elem_type);
      for(size_t i = 0 ; i < array_len; i++){
        auto index = element_size*i;
        std::cout << Extract<double>(val.array.data + index , static_cast<GGufValueType>(val.array.elem_type));
        if(i != array_len - 1) std::cout << ',';
      }
    }else{
      auto element_size = GGufValueSize(val.array.elem_type);
      for(size_t i = 0 ; i < array_len; i++){
        auto index = element_size*i;
        std::cout << Extract<int64_t>(val.array.data + index , static_cast<GGufValueType>(val.array.elem_type));
        if(i != array_len - 1) std::cout << ',';
      }
    }
    std::cout << ']';
  }else if(val.type == GGUF_VALUE_TYPE_STRING){
    std::cout << val.string;
  } else if(val.type == GGUF_VALUE_TYPE_FLOAT32 || val.type == GGUF_VALUE_TYPE_FLOAT64){
    std::cout << Extract<double , GGUF_VALUE_TYPE_FLOAT32 , GGUF_VALUE_TYPE_FLOAT64>(val);
  }else{
    std::cout << Extract<int64_t , GGUF_VALUE_TYPE_UINT8   , GGUF_VALUE_TYPE_INT8    , GGUF_VALUE_TYPE_UINT16  , GGUF_VALUE_TYPE_INT16   , GGUF_VALUE_TYPE_UINT32  , GGUF_VALUE_TYPE_INT32   , GGUF_VALUE_TYPE_BOOL    , GGUF_VALUE_TYPE_UINT64  , GGUF_VALUE_TYPE_INT64   >(val);
  }
}

void debug_print(const GGufTensor& tensor){
  Log("tensor.name",tensor.name);
  Log("  tensor.tensor_type",tensor.tensor_type);
  Log("  tensor.dimension_count",tensor.dimension_count);
  std::cout << "   tensor.dimension ";
  std::cout << "[";
  for(int i = 0 ; i < DIM_ARRAY_MAX_SIZE ; i++){
    std::cout << tensor.dimensions[i];
    if(i != DIM_ARRAY_MAX_SIZE - 1){
      std::cout << ',';
    }
  }
  std::cout << "]" << '\n';
  Log("  tensor.file_offset",tensor.file_offset);
  Log("  tensor.byte_size",tensor.byte_size);
  Log("  tensor.weights_data",reinterpret_cast<uint64_t>(tensor.weights_data));
}

