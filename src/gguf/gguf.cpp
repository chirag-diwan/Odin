#include "../../include/gguf.hpp"

const char * GGufValueName(uint32_t val){
  switch (val) {
    case GGUF_VALUE_TYPE_UINT8:
      return "UINT8";
    case GGUF_VALUE_TYPE_INT8:
      return "INT8";
    case GGUF_VALUE_TYPE_UINT16:
      return "UINT16";
    case GGUF_VALUE_TYPE_INT16:
      return "INT16";
    case GGUF_VALUE_TYPE_UINT32:
      return "UINT32";
    case GGUF_VALUE_TYPE_INT32:
      return "INT32";
    case GGUF_VALUE_TYPE_FLOAT32:
      return "FLOAT32";
    case GGUF_VALUE_TYPE_BOOL:
      return "BOOL";
    case GGUF_VALUE_TYPE_STRING:
      return "STRING";
    case GGUF_VALUE_TYPE_ARRAY:
      return "ARRAY";
    case GGUF_VALUE_TYPE_UINT64:
      return "UINT64";
    case GGUF_VALUE_TYPE_INT64:
      return "INT64";
    case GGUF_VALUE_TYPE_FLOAT64:
      return "FLOAT64";
  }
  return "NULL";
}



size_t GGufValueSize(uint32_t val){
  switch (val) {
    case GGUF_VALUE_TYPE_UINT8:
      return sizeof(uint8_t);
    case GGUF_VALUE_TYPE_INT8:
      return sizeof(int8_t);
    case GGUF_VALUE_TYPE_UINT16:
      return sizeof(uint16_t);
    case GGUF_VALUE_TYPE_INT16:
      return sizeof(int16_t);
    case GGUF_VALUE_TYPE_UINT32:
      return sizeof(uint32_t);
    case GGUF_VALUE_TYPE_INT32:
      return sizeof(int32_t);
    case GGUF_VALUE_TYPE_FLOAT32:
      return sizeof(float);
    case GGUF_VALUE_TYPE_BOOL:
      return sizeof(bool);
    case GGUF_VALUE_TYPE_UINT64:
      return sizeof(uint64_t);
    case GGUF_VALUE_TYPE_INT64:
      return sizeof(int64_t);
    case GGUF_VALUE_TYPE_FLOAT64:
      return sizeof(double);
    default:
      Log(ERROR , "Unsupported size for enum size" , GGufValueName(val));
      std::exit(-1);
  }
}

uint32_t LayerIndex(std::string_view tensor_name) {
  std::string_view prefix = "blk";
  if (tensor_name.size() >= prefix.size() &&
      tensor_name.substr(0, prefix.size()) == prefix) {
    auto start = tensor_name.find(".") + 1;
    auto end   = tensor_name.find(".", start);
    return std::stoi(tensor_name.substr(start, end - start).data());
  }
  Log(ERROR , "Invalid tensor name ", tensor_name);
  std::exit(-1);
}
