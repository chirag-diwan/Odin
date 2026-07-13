#pragma once

#include "./types.hpp"
#include "./logging.hpp"
#include <cstring>
#include <type_traits>

const char * GGufValueName(uint32_t val);

template <typename ret_type , GGufValueType ...valid_pack>
ret_type Extract(const GGufValue& val){
  if(!((val.type == valid_pack) || ...)){
    Log(ERROR , "Unsupported type encountered , the value is" , GGufValueName(val.type) , "wanted" , (GGufValueName(valid_pack),...));
    std::exit(-1);
  } 
  switch (val.type) {
    case GGUF_VALUE_TYPE_UINT8:
      return static_cast<ret_type>(reinterpret_cast<uint8_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_INT8:
      return static_cast<ret_type>(reinterpret_cast<int8_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_UINT16:
      return static_cast<ret_type>(reinterpret_cast<uint16_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_INT16:
      return static_cast<ret_type>(reinterpret_cast<int16_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_UINT32:
      return static_cast<ret_type>(reinterpret_cast<uint32_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_INT32:
      return static_cast<ret_type>(reinterpret_cast<int32_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_FLOAT32:
      return static_cast<ret_type>(reinterpret_cast<float*>(val.data)[0]);
    case GGUF_VALUE_TYPE_BOOL:
      return static_cast<ret_type>(reinterpret_cast<bool*>(val.data)[0]);
    case GGUF_VALUE_TYPE_UINT64:
      return static_cast<ret_type>(reinterpret_cast<uint64_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_INT64:
      return static_cast<ret_type>(reinterpret_cast<int64_t*>(val.data)[0]);
    case GGUF_VALUE_TYPE_FLOAT64:
      return static_cast<ret_type>(reinterpret_cast<double*>(val.data)[0]);
    default:
      Log(ERROR , "Unsupported extraction for %?" , GGufValueName(val.type));
      std::exit(-1);
  }
}


template <typename ret_type , typename T>
ret_type Extract(const T* data , GGufValueType type){
  switch (type) {
    case GGUF_VALUE_TYPE_UINT8:
      return static_cast<ret_type>(reinterpret_cast<const uint8_t*>(data)[0]);
    case GGUF_VALUE_TYPE_INT8:
      return static_cast<ret_type>(reinterpret_cast<const int8_t*>(data)[0]);
    case GGUF_VALUE_TYPE_UINT16:
      return static_cast<ret_type>(reinterpret_cast<const uint16_t*>(data)[0]);
    case GGUF_VALUE_TYPE_INT16:
      return static_cast<ret_type>(reinterpret_cast<const int16_t*>(data)[0]);
    case GGUF_VALUE_TYPE_UINT32:
      return static_cast<ret_type>(reinterpret_cast<const uint32_t*>(data)[0]);
    case GGUF_VALUE_TYPE_INT32:
      return static_cast<ret_type>(reinterpret_cast<const int32_t*>(data)[0]);
    case GGUF_VALUE_TYPE_FLOAT32:
      return static_cast<ret_type>(reinterpret_cast<const float*>(data)[0]);
    case GGUF_VALUE_TYPE_BOOL:
      return static_cast<ret_type>(reinterpret_cast<const bool*>(data)[0]);
    case GGUF_VALUE_TYPE_UINT64:
      return static_cast<ret_type>(reinterpret_cast<const uint64_t*>(data)[0]);
    case GGUF_VALUE_TYPE_INT64:
      return static_cast<ret_type>(reinterpret_cast<const int64_t*>(data)[0]);
    case GGUF_VALUE_TYPE_FLOAT64:
      return static_cast<ret_type>(reinterpret_cast<const double*>(data)[0]);
    default:
      Log(ERROR , "Unsupported extraction for" , GGufValueName(type));
      std::exit(-1);
  }
}



size_t GGufValueSize(uint32_t val);

uint32_t LayerIndex(std::string_view tensor_name) ;

template<typename T>
T read_unaligned(void* ptr){
  static_assert(!std::is_pointer<T>::value);
  T var;
  memcpy(&var,ptr, sizeof(var));
  return var;
}
