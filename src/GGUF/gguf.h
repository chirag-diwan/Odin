#pragma once
#include "../Utils/Utils.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <ggml-cpu.h>
#include <ggml.h>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace Odin {

bool isValidLayerName(std::string_view tensor_name) {
  std::string_view layer_prefix = "blk";
  return tensor_name.size() >= layer_prefix.size() &&
         tensor_name.substr(0, layer_prefix.size()) == layer_prefix;
}

uint32_t getLayerIndex(std::string_view tensor_name) {
  auto start_index = tensor_name.find(".") + 1;
  auto end_index   = tensor_name.find(".", start_index);
  return std::stoi(
      tensor_name.substr(start_index, end_index - start_index).data());
}

enum GgufValueType {
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
};

struct GgufHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

struct GgufArray {
  uint64_t length;
  std::variant<std::nullptr_t, uint8_t*, int8_t*, uint16_t*, int16_t*,
               uint32_t*, int32_t*, float*, uint64_t*, int64_t*, double*, char*,
               bool*, GgufArray*, std::string_view*>
      data = nullptr;
};

struct GgufValue {
  uint64_t length;
  std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
               uint64_t, int64_t, double, char, bool, GgufArray,
               std::string_view>
      data;
};

#ifdef GGUF_DEBUG

void printGgufArray(GgufArray* array_ptr) {
  std::cout << "( size = " << array_ptr->length << " )";
  std::cout << " [";
  std::visit(mix{[&](auto* argument) {
                   if (array_ptr->length > 100) {
                     for (size_t i = 0; i < 100; i++) {
                       std::cout << argument[i] << ',';
                     }
                     std::cout << " first 100 elements out of "
                               << array_ptr->length;
                   } else {
                     for (size_t i = 0; i < array_ptr->length; i++) {
                       std::cout << argument[i] << ',';
                     }
                   }
                 },
                 [](GgufArray* nested_array) {
                   printGgufArray(nested_array);
                 }},
             array_ptr->data);
  std::cout << ']' << '\n';
}

void printGgufValue(GgufValue& value_reference) {
  std::visit(mix{[](auto& argument) {
                   std::cout << argument;
                 },
                 [](GgufArray& array_reference) {
                   printGgufArray(&array_reference);
                 }},
             value_reference.data);
}

#endif

#define DIM_ARRAY_MAX_SIZE 8
struct GgufTensor {
  std::string_view name;
  ggml_type        tensor_type;
  uint32_t         dimension_count;
  int64_t          dimensions[DIM_ARRAY_MAX_SIZE];
  uint64_t         file_offset;
  uint64_t         byte_size;
  GgufTensor*      weights_data;
};

class Model;

class GgufReader {
  friend Model;

private:
  int      file_descriptor;
  uint8_t* mapped_data;
  uint64_t total_size;
  uint64_t remaining_key_values;
  uint64_t remaining_tensors;
  uint64_t current_offset;
  uint64_t data_offset;
  uint64_t byte_alignment;
  uint64_t global_data_offset;
  uint32_t layer_count;

public:
  struct GgufHeader header;

public:
  std::unordered_map<std::string_view, GgufValue> metadata_key_values;
  std::vector<GgufTensor>                         tensors;

private:
  void* getCurrentPositionPointer() {
    return &mapped_data[current_offset];
  }

  inline void advanceOffset(size_t step_size) {
    ERRORIF(current_offset + step_size > total_size, "Size overflow");
    current_offset += step_size;
  }

  std::string_view parseString() {
    auto string_length =
        reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(string_length)));
    std::string_view parsed_string_view(
        static_cast<char*>(getCurrentPositionPointer()), string_length);
    advanceOffset(sizeof(char) * string_length);
    return parsed_string_view;
  }

  GgufArray parseArray() {
    auto element_type =
        reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(element_type)));

    auto element_count =
        reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(element_count)));

    GgufArray parsed_array;
    parsed_array.length = element_count;

    switch (element_type) {
    case GGUF_VALUE_TYPE_UINT8: {
      parsed_array.data =
          reinterpret_cast<uint8_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(uint8_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_INT8: {
      parsed_array.data =
          reinterpret_cast<int8_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(int8_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_UINT16: {
      parsed_array.data =
          reinterpret_cast<uint16_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(uint16_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_INT16: {
      parsed_array.data =
          reinterpret_cast<int16_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(int16_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_UINT32: {
      parsed_array.data =
          reinterpret_cast<uint32_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(uint32_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_INT32: {
      parsed_array.data =
          reinterpret_cast<int32_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(int32_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_FLOAT32: {
      parsed_array.data = reinterpret_cast<float*>(getCurrentPositionPointer());
      advanceOffset(sizeof(float) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_BOOL: {
      parsed_array.data = reinterpret_cast<bool*>(getCurrentPositionPointer());
      advanceOffset(sizeof(bool) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_STRING: {
      auto string_pointers = new std::string_view[element_count];
      for (size_t i = 0; i < element_count; i++) {
        string_pointers[i] = parseString();
      }
      parsed_array.data = string_pointers;
      break;
    }
    case GGUF_VALUE_TYPE_ARRAY: {
      auto array_pointers = new GgufArray[element_count];
      for (size_t i = 0; i < element_count; i++) {
        array_pointers[i] = parseArray();
      }
      parsed_array.data = array_pointers;
      break;
    }
    case GGUF_VALUE_TYPE_UINT64: {
      parsed_array.data =
          reinterpret_cast<uint64_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(uint64_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_INT64: {
      parsed_array.data =
          reinterpret_cast<int64_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(int64_t) * element_count);
      break;
    }
    case GGUF_VALUE_TYPE_FLOAT64: {
      parsed_array.data =
          reinterpret_cast<double*>(getCurrentPositionPointer());
      advanceOffset(sizeof(double) * element_count);
      break;
    }
    default:
      ERROR_AND_EXIT("Invalid type in array");
    }
    return parsed_array;
  }

  void parseKeyValue() {
    auto metadata_key = parseString();
    auto value_type =
        reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(value_type)));

    GgufValue parsed_value;

    switch (value_type) {
    case GGUF_VALUE_TYPE_UINT8:
      parsed_value.data =
          reinterpret_cast<uint8_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(uint8_t));
      break;

    case GGUF_VALUE_TYPE_INT8:
      parsed_value.data =
          reinterpret_cast<int8_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(int8_t));
      break;

    case GGUF_VALUE_TYPE_UINT16:
      parsed_value.data =
          reinterpret_cast<uint16_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(uint16_t));
      break;

    case GGUF_VALUE_TYPE_INT16:
      parsed_value.data =
          reinterpret_cast<int16_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(int16_t));
      break;

    case GGUF_VALUE_TYPE_UINT32:
      parsed_value.data =
          reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(uint32_t));
      break;

    case GGUF_VALUE_TYPE_INT32:
      parsed_value.data =
          reinterpret_cast<int32_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(int32_t));
      break;

    case GGUF_VALUE_TYPE_FLOAT32:
      parsed_value.data =
          reinterpret_cast<float*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(float));
      break;

    case GGUF_VALUE_TYPE_BOOL:
      parsed_value.data =
          reinterpret_cast<bool*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(bool));
      break;

    case GGUF_VALUE_TYPE_STRING:
      parsed_value.data = parseString();
      break;

    case GGUF_VALUE_TYPE_ARRAY:
      parsed_value.data = parseArray();
      break;

    case GGUF_VALUE_TYPE_UINT64:
      parsed_value.data =
          reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(uint64_t));
      break;

    case GGUF_VALUE_TYPE_INT64:
      parsed_value.data =
          reinterpret_cast<int64_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(int64_t));
      break;

    case GGUF_VALUE_TYPE_FLOAT64:
      parsed_value.data =
          reinterpret_cast<double*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(double));
      break;

    default:
      ERROR_AND_EXIT("Unknown Type");
    }
    metadata_key_values[metadata_key] = std::move(parsed_value);
  }

  inline void cleanupResources() {
    munmap(mapped_data, total_size);
    std::function<void(GgufArray&)> recursive_clean;
    recursive_clean = [&](GgufArray& current_array) {
      std::visit(mix{[](auto&) {
                     },
                     [](std::string_view* string_argument) {
                       delete[] string_argument;
                     },
                     [&](GgufArray* array_argument) {
                       for (size_t i = 0; i < current_array.length; ++i) {
                         recursive_clean(array_argument[i]);
                       }
                       delete[] array_argument;
                     }},
                 current_array.data);
    };
  }

public:
  GgufReader() = default;

  void openFile(const char* filepath) {
    int opened_descriptor = open(filepath, O_RDONLY);
    ERRORIF(opened_descriptor == -1, "Not a valid file descriptor for ",
            filepath);
    struct stat file_statistics;
    ERRORIF(fstat(opened_descriptor, &file_statistics) == -1,
            "Unable to get file stats for ", filepath);
    void* memory_mapped_pointer = mmap(NULL, file_statistics.st_size, PROT_READ,
                                       MAP_PRIVATE, opened_descriptor, 0);
    ERRORIF(memory_mapped_pointer == MAP_FAILED, "Mapping failed for ",
            filepath);

    this->file_descriptor = opened_descriptor;
    this->mapped_data     = reinterpret_cast<uint8_t*>(memory_mapped_pointer);
    this->total_size      = file_statistics.st_size;
    this->current_offset  = 0;
    this->data_offset     = 0;
    this->byte_alignment  = 32;
    this->global_data_offset = 0;
    this->layer_count        = 0;
  }

  void parseHeader() {
    ERRORIF(current_offset != 0, "Offset is not zero on the first call");

    header =
        reinterpret_cast<struct GgufHeader*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(header)));
  }

  void parseAllKeyValues() {
    for (size_t i = 0; i < header.metadata_kv_count; ++i) {
      parseKeyValue();
    }

    if (metadata_key_values.find("general.alignment") !=
        metadata_key_values.end()) {
      std::visit(mix{[](auto& unhandled_type) {
                       ERROR_AND_EXIT("Invalid key type for general.alignment");
                     },

                     [this](uint32_t& alignment_value) {
                       this->byte_alignment = alignment_value;
                     },

                     [this](uint64_t& alignment_value) {
                       this->byte_alignment = alignment_value;
                     }},

                 metadata_key_values["general.alignment"].data);
    }
#ifdef GGUF_DEBUG
    for (auto& [metadata_key, metadata_value] : metadata_key_values) {
      std::cout << "\n" << metadata_key << "    ---->   ";
      printGgufValue(metadata_value);
    }
#endif
  }

  void parseAllTensors(ggml_context* weight_context = nullptr) {

    std::vector<uint32_t> layer_indices(32);

    for (size_t i = 0; i < header.tensor_count; ++i) {
      GgufTensor current_tensor;
      auto       tensor_name = parseString();

      if (isValidLayerName(tensor_name)) {
        auto layer_index = getLayerIndex(tensor_name);
        if (std::find(layer_indices.begin(), layer_indices.end(),
                      layer_index) == layer_indices.end()) {
          layer_indices.push_back(layer_index);
          layer_count++;
        }
      }

      auto dimension_count =
          reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(dimension_count)));

      std::array<int64_t, DIM_ARRAY_MAX_SIZE> tensor_dimensions = {1};

      for (size_t j = 0; j < dimension_count; j++) {
        tensor_dimensions[j] =
            reinterpret_cast<int64_t*>(getCurrentPositionPointer())[0];
        advanceOffset(sizeof(int64_t));
      }

      auto ggml_data_type =
          reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(ggml_data_type)));

      auto tensor_offset =
          reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(tensor_offset)));

      current_tensor.name            = tensor_name;
      current_tensor.file_offset     = tensor_offset;
      current_tensor.tensor_type     = static_cast<ggml_type>(ggml_data_type);
      current_tensor.dimension_count = dimension_count;

      for (int i = 0; i < DIM_ARRAY_MAX_SIZE; i++) {
        current_tensor.dimensions[i] = tensor_dimensions[i];
      }

      current_tensor.byte_size = 1;
      for (uint8_t i = 0; i < current_tensor.dimension_count; ++i) {
        current_tensor.byte_size *= current_tensor.dimensions[i];
      }
      const auto block_size = ggml_blck_size(current_tensor.tensor_type);
      ERRORIF(current_tensor.byte_size % block_size != 0,
              "Number of elements in tensor ", current_tensor.name,
              " is not a multiple of block size ", block_size);
      current_tensor.byte_size = current_tensor.byte_size *
                                 ggml_type_size(current_tensor.tensor_type) /
                                 block_size;
      tensors.emplace_back(current_tensor);
    }
    this->global_data_offset =
        (this->current_offset + this->byte_alignment - 1) &
        ~(this->byte_alignment - 1);

#ifdef GGUF_DEBUG
    for (const auto& tensor_item : tensors) {
      std::cout << "\nTensor {\n";

      std::cout << "  Name   : " << tensor_item.name << "\n";
      std::cout << "  NDims  : " << tensor_item.dimension_count << "\n";

      std::cout << "  Shape  : [";
      for (int i = 0; i < tensor_item.dimension_count; ++i) {
        std::cout << tensor_item.dimensions[i];
        if (i != tensor_item.dimension_count - 1)
          std::cout << ", ";
      }
      std::cout << "]\n";

      std::cout << "  Offset : " << tensor_item.file_offset << "\n";
      std::cout << "  Size(in bytes) : " << tensor_item.byte_size << "\n";

      std::cout << "  Type   : " << ggml_type_name(tensor_item.tensor_type)
                << " (" << tensor_item.tensor_type << ")\n";

      std::cout << "}";
    }
#endif
  }

  ~GgufReader() {
    cleanupResources();
  }
};
}; // namespace Odin
