#pragma once
#include "../Utils/Utils.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <memory>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#define GGUF_NONE 0

// GGUF_TYPE_Q4_2 = 4, support has been removed
// GGUF_TYPE_Q4_3 (5) support has been removed
enum gguf_tensor_type {
  GGUF_TYPE_F32     = 0,
  GGUF_TYPE_F16     = 1,
  GGUF_TYPE_Q4_0    = 2,
  GGUF_TYPE_Q4_1    = 3,
  GGUF_TYPE_Q5_0    = 6,
  GGUF_TYPE_Q5_1    = 7,
  GGUF_TYPE_Q8_0    = 8,
  GGUF_TYPE_Q8_1    = 9,
  GGUF_TYPE_Q2_K    = 10,
  GGUF_TYPE_Q3_K    = 11,
  GGUF_TYPE_Q4_K    = 12,
  GGUF_TYPE_Q5_K    = 13,
  GGUF_TYPE_Q6_K    = 14,
  GGUF_TYPE_Q8_K    = 15,
  GGUF_TYPE_IQ2_XXS = 16,
  GGUF_TYPE_IQ2_XS  = 17,
  GGUF_TYPE_IQ3_XXS = 18,
  GGUF_TYPE_IQ1_S   = 19,
  GGUF_TYPE_IQ4_NL  = 20,
  GGUF_TYPE_IQ3_S   = 21,
  GGUF_TYPE_IQ2_S   = 22,
  GGUF_TYPE_IQ4_XS  = 23,
  GGUF_TYPE_I8      = 24,
  GGUF_TYPE_I16     = 25,
  GGUF_TYPE_I32     = 26,
  GGUF_TYPE_I64     = 27,
  GGUF_TYPE_F64     = 28,
  GGUF_TYPE_IQ1_M   = 29,
  GGUF_TYPE_BF16    = 30,
  GGUF_TYPE_COUNT,
};

#ifdef GGUF_DEBUG
const char* gguf_tensor_type_to_string(gguf_tensor_type type) {
  switch (type) {
  case GGUF_TYPE_F32:
    return "F32";
  case GGUF_TYPE_F16:
    return "F16";
  case GGUF_TYPE_Q4_0:
    return "Q4_0";
  case GGUF_TYPE_Q4_1:
    return "Q4_1";
  case GGUF_TYPE_Q5_0:
    return "Q5_0";
  case GGUF_TYPE_Q5_1:
    return "Q5_1";
  case GGUF_TYPE_Q8_0:
    return "Q8_0";
  case GGUF_TYPE_Q8_1:
    return "Q8_1";
  case GGUF_TYPE_Q2_K:
    return "Q2_K";
  case GGUF_TYPE_Q3_K:
    return "Q3_K";
  case GGUF_TYPE_Q4_K:
    return "Q4_K";
  case GGUF_TYPE_Q5_K:
    return "Q5_K";
  case GGUF_TYPE_Q6_K:
    return "Q6_K";
  case GGUF_TYPE_Q8_K:
    return "Q8_K";
  case GGUF_TYPE_IQ2_XXS:
    return "IQ2_XXS";
  case GGUF_TYPE_IQ2_XS:
    return "IQ2_XS";
  case GGUF_TYPE_IQ3_XXS:
    return "IQ3_XXS";
  case GGUF_TYPE_IQ1_S:
    return "IQ1_S";
  case GGUF_TYPE_IQ4_NL:
    return "IQ4_NL";
  case GGUF_TYPE_IQ3_S:
    return "IQ3_S";
  case GGUF_TYPE_IQ2_S:
    return "IQ2_S";
  case GGUF_TYPE_IQ4_XS:
    return "IQ4_XS";
  case GGUF_TYPE_I8:
    return "I8";
  case GGUF_TYPE_I16:
    return "I16";
  case GGUF_TYPE_I32:
    return "I32";
  case GGUF_TYPE_I64:
    return "I64";
  case GGUF_TYPE_F64:
    return "F64";
  case GGUF_TYPE_IQ1_M:
    return "IQ1_M";
  case GGUF_TYPE_BF16:
    return "BF16";
  default:
    return "UNKNOWN";
  }
}
#endif

enum gguf_value_type {
  // The value is a 8-bit unsigned integer.
  GGUF_VALUE_TYPE_UINT8 = 0,
  // The value is a 8-bit signed integer.
  GGUF_VALUE_TYPE_INT8 = 1,
  // The value is a 16-bit unsigned little-endian integer.
  GGUF_VALUE_TYPE_UINT16 = 2,
  // The value is a 16-bit signed little-endian integer.
  GGUF_VALUE_TYPE_INT16 = 3,
  // The value is a 32-bit unsigned little-endian integer.
  GGUF_VALUE_TYPE_UINT32 = 4,
  // The value is a 32-bit signed little-endian integer.
  GGUF_VALUE_TYPE_INT32 = 5,
  // The value is a 32-bit IEEE754 floating point number.
  GGUF_VALUE_TYPE_FLOAT32 = 6,
  // The value is a boolean.
  // 1-byte value where 0 is false and 1 is true.
  // Anything else is invalid, and should be treated as either the model
  // being invalid or the reader being buggy.
  GGUF_VALUE_TYPE_BOOL = 7,
  // The value is a UTF-8 non-null-terminated string, with length prepended.
  GGUF_VALUE_TYPE_STRING = 8,
  // The value is an array of other values, with the length and type
  // prepended. Arrays can be nested, and the length of the array is the
  // number of elements in the array, not the number of bytes.
  GGUF_VALUE_TYPE_ARRAY = 9,
  // The value is a 64-bit unsigned little-endian integer.
  GGUF_VALUE_TYPE_UINT64 = 10,
  // The value is a 64-bit signed little-endian integer.
  GGUF_VALUE_TYPE_INT64 = 11,
  // The value is a 64-bit IEEE754 floating point number.
  GGUF_VALUE_TYPE_FLOAT64 = 12,
};

struct gguf_header {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // or just compare to 119585722
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec.
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
};

struct gguf_array {
    uint64_t len;
    std::variant<std::nullptr_t, uint8_t*, int8_t*, uint16_t*, int16_t*,
                 uint32_t*, int32_t*, float*, uint64_t*, int64_t*, double*,
                 char*, bool*, gguf_array*, std::string_view*>
        data = nullptr;
};

struct gguf_value {
    uint64_t len;
    std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
                 uint64_t, int64_t, double, char, bool, gguf_array,
                 std::string_view>
        data;
    ;
};

#ifdef GGUF_DEBUG

void Print_gguf_arr(gguf_array* arr) {
  std::cout << "( size = " << arr->len << " )";
  std::cout << " [";
  std::visit(mix{[&](auto* arg) {
                   if (arr->len > 100) {
                     for (size_t i = 0; i < 100; i++) {
                       std::cout << arg[i] << ',';
                     }
                     std::cout << " first 100 elements out of " << arr->len;
                   } else {
                     for (size_t i = 0; i < arr->len; i++) {
                       std::cout << arg[i] << ',';
                     }
                   }
                 },
                 [](gguf_array* a) { Print_gguf_arr(a); }},
             arr->data);
  std::cout << ']' << '\n';
}

void Print_gguf_val(gguf_value& v) {
  std::visit(mix{[](auto& arg) { std::cout << arg; },
                 [](gguf_array& arr) { Print_gguf_arr(&arr); }},
             v.data);
}

#endif

typedef struct {
    const char* name;
    size_t      namelen;
    uint32_t    type;         // Tensor type (enum gguf_tensor_type).
    uint32_t    ndim;         // Number of dimensions of the tensor.
    uint64_t*   dim;          // Dimensions (Eg. [512, 1024, 1, 1]).
    uint64_t    offset;       // Offset from start of file.
    uint64_t    bsize;        // Total size in bytes.
    uint64_t    num_weights;  // Total number of parameters.
    uint8_t*    weights_data; // Pointer to the mmaped file.
} gguf_tensor;

class GGUF {
  private:
    int                fd;
    uint8_t*           data;
    uint64_t           size;
    struct gguf_header header;
    uint64_t           left_kv;
    uint64_t           left_tensors;
    uint64_t           off;
    uint64_t           data_off;
    uint64_t           alignment;

  public:
    std::unordered_map<std::string_view, gguf_value> metadata_kv;

  private:
    void* pos() {
      return &data[off];
    }

    inline void increment(size_t s) {
      ERRORIF(off + s > size, "Size overflow");
      off += s;
    }

    std::string_view parsestring() {
      auto str_len = reinterpret_cast<uint64_t*>(pos())[0];
      increment(sizeof(decltype(str_len)));
      std::string_view sv(static_cast<char*>(pos()), str_len);
      increment(sizeof(char) * str_len);
      return sv;
    }

    // Expects gguf array at pos
    gguf_array parsearray() {
      auto elem_type = reinterpret_cast<uint32_t*>(pos())[0];
      increment(sizeof(decltype(elem_type)));

      auto n_elem = reinterpret_cast<uint64_t*>(pos())[0];
      increment(sizeof(decltype(n_elem)));

      gguf_array arr;
      arr.len = n_elem;

      switch (elem_type) {
      case GGUF_VALUE_TYPE_UINT8: {
        arr.data = reinterpret_cast<uint8_t*>(pos());
        increment(sizeof(uint8_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_INT8: {
        arr.data = reinterpret_cast<int8_t*>(pos());
        increment(sizeof(int8_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_UINT16: {
        arr.data = reinterpret_cast<uint16_t*>(pos());
        increment(sizeof(uint16_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_INT16: {
        arr.data = reinterpret_cast<int16_t*>(pos());
        increment(sizeof(int16_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_UINT32: {
        arr.data = reinterpret_cast<uint32_t*>(pos());
        increment(sizeof(uint32_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_INT32: {
        arr.data = reinterpret_cast<int32_t*>(pos());
        increment(sizeof(int32_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_FLOAT32: {
        arr.data = reinterpret_cast<float*>(pos());
        increment(sizeof(float) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_BOOL: {
        arr.data = reinterpret_cast<bool*>(pos());
        increment(sizeof(bool) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_STRING: {
        auto strs = new std::string_view[n_elem];
        for (size_t i = 0; i < n_elem; i++) {
          strs[i] = parsestring();
        }
        arr.data = strs;
        break;
      }
      case GGUF_VALUE_TYPE_ARRAY: {
        auto ptrs = new gguf_array[n_elem];
        for (size_t i = 0; i < n_elem; i++) {
          ptrs[i] = parsearray();
        }
        arr.data = ptrs;
        break;
      }
      case GGUF_VALUE_TYPE_UINT64: {
        arr.data = reinterpret_cast<uint64_t*>(pos());
        increment(sizeof(uint64_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_INT64: {
        arr.data = reinterpret_cast<int64_t*>(pos());
        increment(sizeof(int64_t) * n_elem);
        break;
      }
      case GGUF_VALUE_TYPE_FLOAT64: {
        arr.data = reinterpret_cast<double*>(pos());
        increment(sizeof(double) * n_elem);
        break;
      }
      default:
        ERROR_AND_EXIT("Invalid type in array");
      }
      return arr;
    }

    // parses one key value pair , expects the key value pair starting to be at
    // pos()
    void parse_key_value() {
      auto key  = parsestring();
      auto type = reinterpret_cast<uint32_t*>(pos())[0];
      increment(sizeof(decltype(type)));

      gguf_value value;

      switch (type) {
      case GGUF_VALUE_TYPE_UINT8:
        value.data = reinterpret_cast<uint8_t*>(pos())[0];
        increment(sizeof(uint8_t));
        break;

      case GGUF_VALUE_TYPE_INT8:
        value.data = reinterpret_cast<int8_t*>(pos())[0];
        increment(sizeof(int8_t));
        break;

      case GGUF_VALUE_TYPE_UINT16:
        value.data = reinterpret_cast<uint16_t*>(pos())[0];
        increment(sizeof(uint16_t));
        break;

      case GGUF_VALUE_TYPE_INT16:
        value.data = reinterpret_cast<int16_t*>(pos())[0];
        increment(sizeof(int16_t));
        break;

      case GGUF_VALUE_TYPE_UINT32:
        value.data = reinterpret_cast<uint32_t*>(pos())[0];
        increment(sizeof(uint32_t));
        break;

      case GGUF_VALUE_TYPE_INT32:
        value.data = reinterpret_cast<int32_t*>(pos())[0];
        increment(sizeof(int32_t));
        break;

      case GGUF_VALUE_TYPE_FLOAT32:
        value.data = reinterpret_cast<float*>(pos())[0];
        increment(sizeof(float));
        break;

      case GGUF_VALUE_TYPE_BOOL:
        value.data = reinterpret_cast<bool*>(pos())[0];
        increment(sizeof(bool));
        break;

      case GGUF_VALUE_TYPE_STRING:
        value.data = parsestring();
        break;

      case GGUF_VALUE_TYPE_ARRAY:
        value.data = parsearray();
        break;

      case GGUF_VALUE_TYPE_UINT64:
        value.data = reinterpret_cast<uint64_t*>(pos())[0];
        increment(sizeof(uint64_t));
        break;

      case GGUF_VALUE_TYPE_INT64:
        value.data = reinterpret_cast<int64_t*>(pos())[0];
        increment(sizeof(int64_t));
        break;

      case GGUF_VALUE_TYPE_FLOAT64:
        value.data = reinterpret_cast<double*>(pos())[0];
        increment(sizeof(double));
        break;

      default:
        ERROR_AND_EXIT("Unknown Type");
      }
      metadata_kv[key] = std::move(value);
    }

    inline void clean() {
      munmap(data, size);
      std::function<void(gguf_array&)> clean;
      clean = [&](gguf_array& val) {
        std::visit(mix{[](auto&) {},
                       [](std::string_view* arg) { delete[] arg; },
                       [&](gguf_array* arg) {
                         for (size_t i = 0; i < val.len; ++i) {
                           clean(arg[i]);
                         }
                         delete[] arg;
                       }},
                   val.data);
      };
    }

  public:
    GGUF() = default;

    void OpenFile(const char* filepath) {
      int fd = open(filepath, O_RDONLY);
      ERRORIF(fd == -1, "Not a valid file descriptor for ", filepath);
      struct stat file_stat;
      ERRORIF(fstat(fd, &file_stat) == -1, "Unable to get file stats for ",
              filepath);
      void* map_ptr =
          mmap(NULL, file_stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      ERRORIF(map_ptr == MAP_FAILED, "Mapping failed for ", filepath);

      this->fd        = fd;
      this->data      = reinterpret_cast<uint8_t*>(map_ptr);
      this->size      = file_stat.st_size;
      this->off       = 0;
      this->data_off  = 0;
      this->alignment = 32;
    }

    void ParseHeader() {
      ERRORIF(off != 0, "Offset is not zero on the first call");

      header = reinterpret_cast<struct gguf_header*>(pos())[0];
      increment(sizeof(decltype(header)));
    }

    void ParseKeyValue() {
      for (size_t i = 0; i < header.metadata_kv_count; ++i) {
        parse_key_value();
      }
#ifdef GGUF_DEBUG
      for (auto& [key, v] : metadata_kv) {
        std::cout << "\n" << key << "   ---->   ";
        Print_gguf_val(v);
      }
#endif
    }

    void ParseTensors() {

#ifdef GGUF_DEBUG
      std::cout << "Tensor -> Number of dimensions -> dimensions -> Offset \n";
#endif
#define DIM_ARRAY_MAX_SIZE 8
      for (size_t i = 0; i < header.tensor_count; ++i) {
        auto tensor_name = parsestring();
        auto n_dim       = reinterpret_cast<uint32_t*>(pos())[0];
        increment(sizeof(decltype(n_dim)));

        std::array<int64_t, DIM_ARRAY_MAX_SIZE> dim = {0};

        for (size_t j = 0; j < n_dim; j++) {
          dim[j] = reinterpret_cast<int64_t*>(pos())[0];
          increment(sizeof(int64_t));
        }

        auto type = reinterpret_cast<uint32_t*>(pos())[0];
        increment(sizeof(decltype(type)));

        auto offset = reinterpret_cast<uint64_t*>(pos())[0];
        increment(sizeof(decltype(offset)));

#ifdef GGUF_DEBUG
        std::cout << "Tensor {\n";

        std::cout << "  Name   : " << tensor_name << "\n";
        std::cout << "  NDims  : " << n_dim << "\n";

        std::cout << "  Shape  : [";
        for (int i = 0; i < n_dim; ++i) {
          std::cout << dim[i];
          if (i != n_dim - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  Offset : " << offset << "\n";

        std::cout << "  Type   : "
                  << gguf_tensor_type_to_string(
                         static_cast<gguf_tensor_type>(type))
                  << " (" << type << ")\n";

        std::cout << "}\n";
#endif
      }
    }

    ~GGUF() {
      clean();
    }
};
