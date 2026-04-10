#pragma once
#include "../Utils/Utils.h"

#include <array>
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
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace Odin {

struct ModelBlock {
  // Exists in: PreNorm, PostNorm.
  // Bias exists if: NormalizationType == LayerNorm
  struct ggml_tensor* attn_norm_w = nullptr;
  struct ggml_tensor* attn_norm_b = nullptr;

  // Exists in: PreNorm, PostNorm. Does NOT exist in Parallel (reuses
  // attn_norm).
  struct ggml_tensor* ffn_norm_w = nullptr;
  struct ggml_tensor* ffn_norm_b = nullptr;

  // Biases exist if: BiasTopology == Biased OR AttentionOnly
  // Layout: Separated (Llama, Qwen)
  struct ggml_tensor* attn_q_w = nullptr;
  struct ggml_tensor* attn_q_b = nullptr;
  struct ggml_tensor* attn_k_w = nullptr;
  struct ggml_tensor* attn_k_b = nullptr;
  struct ggml_tensor* attn_v_w = nullptr;
  struct ggml_tensor* attn_v_b = nullptr;

  // Layout: PackedQKV (Falcon, Bloom)
  struct ggml_tensor* attn_qkv_w = nullptr;
  struct ggml_tensor* attn_qkv_b = nullptr;

  // Layout: PackedKV (Some MQA variants)
  // Uses attn_q_w above, plus these combined KV weights
  struct ggml_tensor* attn_kv_w = nullptr;
  struct ggml_tensor* attn_kv_b = nullptr;

  // Always exists. Bias depends on BiasTopology.
  struct ggml_tensor* attn_output_w = nullptr;
  struct ggml_tensor* attn_output_b = nullptr;

  // Topology: Standard (GPT-2, Bloom)
  // ffn_up expands the dimension, ffn_down reduces it back.
  // Bias depends on BiasTopology.
  struct ggml_tensor* ffn_up_w   = nullptr;
  struct ggml_tensor* ffn_up_b   = nullptr;
  struct ggml_tensor* ffn_down_w = nullptr;
  struct ggml_tensor* ffn_down_b = nullptr;

  // Topology: Gated / SwiGLU (Llama, Qwen, Mistral)
  // Gate is the activation branch, Up is the linear branch. Down is the output.
  // Modern architectures rarely use biases here, but the engine must support
  // the possibility.
  struct ggml_tensor* ffn_gate_w = nullptr;
  struct ggml_tensor* ffn_gate_b = nullptr;
  // Uses ffn_up_w and ffn_down_w from above.

  ModelBlock() = default;
};

struct ModelGlobalTensors {
  struct ggml_tensor* token_embd_weights  = nullptr;
  struct ggml_tensor* output_norm_weights = nullptr;
  struct ggml_tensor* output_weights      = nullptr;
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
};

void MapTensorToBlock(std::string_view& tensor_name, ggml_tensor* tensor,
                      ModelBlock& block) {

  // --- Normalization ---
  if (tensor_name.find(".attn_norm.weight") != std::string::npos)
    block.attn_norm_w = tensor;
  else if (tensor_name.find(".ffn_norm.weight") != std::string::npos)
    block.ffn_norm_w = tensor;

  // --- Attention QKV ---
  else if (tensor_name.find(".attn_q.weight") != std::string::npos)
    block.attn_q_w = tensor;
  else if (tensor_name.find(".attn_q.bias") != std::string::npos)
    block.attn_q_b = tensor;
  else if (tensor_name.find(".attn_k.weight") != std::string::npos)
    block.attn_k_w = tensor;
  else if (tensor_name.find(".attn_k.bias") != std::string::npos)
    block.attn_k_b = tensor;
  else if (tensor_name.find(".attn_v.weight") != std::string::npos)
    block.attn_v_w = tensor;
  else if (tensor_name.find(".attn_v.bias") != std::string::npos)
    block.attn_v_b = tensor;
  else if (tensor_name.find(".attn_qkv.weight") != std::string::npos)
    block.attn_qkv_w = tensor;

  // --- Attention Output ---
  else if (tensor_name.find(".attn_output.weight") != std::string::npos)
    block.attn_output_w = tensor;

  // --- FFN (SwiGLU specific to Qwen/Llama) ---
  else if (tensor_name.find(".ffn_gate.weight") != std::string::npos)
    block.ffn_gate_w = tensor;
  else if (tensor_name.find(".ffn_up.weight") != std::string::npos)
    block.ffn_up_w = tensor;
  else if (tensor_name.find(".ffn_down.weight") != std::string::npos)
    block.ffn_down_w = tensor;
}

uint32_t LayerIndex(std::string_view tensor_name) {
  std::string_view prefix = "blk";
  if (tensor_name.size() >= prefix.size() &&
      tensor_name.substr(0, prefix.size()) == prefix) {
    auto start = tensor_name.find(".") + 1;
    auto end   = tensor_name.find(".", start);
    return std::stoi(tensor_name.substr(start, end - start).data());
  }
  ERROR_AND_EXIT("Invalid tensor name ", tensor_name);
  return -1; // not gonna reach here
}

struct GGufHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

struct GGufArray {
  uint64_t length;
  std::variant<std::nullptr_t, uint8_t*, int8_t*, uint16_t*, int16_t*,
               uint32_t*, int32_t*, float*, uint64_t*, int64_t*, double*, char*,
               bool*, GGufArray*, std::string_view*>
      data = nullptr;
};

struct GgufValue {
  uint64_t length;
  std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float,
               uint64_t, int64_t, double, char, bool, GGufArray,
               std::string_view>
      data;
};

#ifdef GGUF_DEBUG
void printGgufArray(GGufArray* array_ptr) {
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
                 [](GGufArray* nested_array) {
                   printGgufArray(nested_array);
                 }},
             array_ptr->data);
  std::cout << ']' << '\n';
}

void printGgufValue(GgufValue& value_reference) {
  std::visit(mix{[](auto& argument) {
                   std::cout << argument;
                 },
                 [](GGufArray& array_reference) {
                   printGgufArray(&array_reference);
                 }},
             value_reference.data);
}

const char* GetTypeName(GGufValueType t) {
  switch (t) {
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

class GGufReader {
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

  uint32_t model_block_count;
  uint32_t model_head_count_kv;
  uint32_t model_head_count;
  uint32_t model_embedding_length;
  float    model_rope_freq_base;

public:
  struct GGufHeader header;

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

  GGufArray parseArray() {
    auto element_type =
        reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(element_type)));

    auto element_count =
        reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(element_count)));

    GGufArray parsed_array;
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
      auto array_pointers = new GGufArray[element_count];
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
    std::function<void(GGufArray&)> recursive_clean;
    recursive_clean = [&](GGufArray& current_array) {
      std::visit(mix{[](auto&) {
                     },
                     [](std::string_view* string_argument) {
                       delete[] string_argument;
                     },
                     [&](GGufArray* array_argument) {
                       for (size_t i = 0; i < current_array.length; ++i) {
                         recursive_clean(array_argument[i]);
                       }
                       delete[] array_argument;
                     }},
                 current_array.data);
    };
    for (auto& [metadata_key, metadata_value] : metadata_key_values) {
      std::visit(mix{
                     [](auto&) {
                     },
                     [&](GGufArray& gguf_array) {
                       recursive_clean(gguf_array);
                     },
                 },
                 metadata_value.data);
    }
  }

public:
  GGufReader() = default;

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
  }

  void parseHeader() {
    ERRORIF(current_offset != 0, "Offset is not zero on the first call");

    header =
        reinterpret_cast<struct GGufHeader*>(getCurrentPositionPointer())[0];
    advanceOffset(sizeof(decltype(header)));
#ifdef GGUF_DEBUG
    std::cout << "Magic " << header.magic << '\n';
    std::cout << "Version " << header.version << '\n';
    std::cout << "Tensor count " << header.tensor_count << '\n';
    std::cout << "Key Value count " << header.metadata_kv_count << '\n';
#endif
  }

  void parseAllKeyValues() {
    for (size_t i = 0; i < header.metadata_kv_count; ++i) {
      parseKeyValue();
    }

    // Extract some important key value pair
    if (metadata_key_values.find("general.alignment") !=
        metadata_key_values.end()) {
      std::visit(mix{[](auto&) {
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

    for (auto& [metadata_key, metadata_value] : metadata_key_values) {
      if (metadata_key.find("block_count") != static_cast<size_t>(-1)) {
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT(
                             "Invalid key type for general.alignment");
                       },

                       [this](uint32_t& alignment_value) {
                         this->model_block_count = alignment_value;
                       },

                       [this](uint64_t& alignment_value) {
                         this->model_block_count = alignment_value;
                       }},
                   metadata_value.data);
      } else if (metadata_key.find("head_count_kv") !=
                 static_cast<size_t>(-1)) {
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT("Invalid key type for head_count_kv");
                       },

                       [this](uint32_t& alignment_value) {
                         this->model_head_count_kv = alignment_value;
                       },

                       [this](uint64_t& alignment_value) {
                         this->model_head_count_kv = alignment_value;
                       }},
                   metadata_value.data);
      } else if (metadata_key.find("head_count") != static_cast<size_t>(-1)) {
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT("Invalid key type for head_count");
                       },

                       [this](uint32_t& alignment_value) {
                         this->model_head_count = alignment_value;
                       },

                       [this](uint64_t& alignment_value) {
                         this->model_head_count = alignment_value;
                       }},

                   metadata_value.data);
      } else if (metadata_key.find("embedding_length") !=
                 static_cast<size_t>(-1)) {
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT(
                             "Invalid key type for embedding_length");
                       },

                       [this](uint32_t& alignment_value) {
                         this->model_embedding_length = alignment_value;
                       },

                       [this](uint64_t& alignment_value) {
                         this->model_embedding_length = alignment_value;
                       }},
                   metadata_value.data);
      } else if (metadata_key.find("rope.freq_base") !=
                 static_cast<size_t>(-1)) {
        std::visit(mix{
                       [](auto&) {
                         ERROR_AND_EXIT("Invalid key type for rope.freq_base");
                       },

                       [this](float& alignment_value) {
                         this->model_rope_freq_base = alignment_value;
                       },

                   },
                   metadata_value.data);
      }

      // rope.freq_base
    }

#ifdef GGUF_DEBUG
    for (auto& [metadata_key, metadata_value] : metadata_key_values) {
      std::cout << "\n" << metadata_key << "    ---->   ";
      printGgufValue(metadata_value);
    }
#endif
  }

  void parseAllTensors(std::vector<ModelBlock>& blocks,
                       ModelGlobalTensors&      global_tensor,
                       ggml_context*            weight_context = nullptr) {
    blocks.resize(model_block_count); // initialize empty blocks

    for (size_t i = 0; i < header.tensor_count; ++i) {
      auto tensor_name = parseString();

      auto dimension_count =
          reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(dimension_count)));

      std::array<int64_t, DIM_ARRAY_MAX_SIZE> tensor_dimensions = {1};
      for (size_t j = 0; j < dimension_count; j++) {
        tensor_dimensions[j] =
            reinterpret_cast<int64_t*>(getCurrentPositionPointer())[0];
        advanceOffset(sizeof(int64_t));
      }

      auto ggml_data_type = static_cast<ggml_type>(
          reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0]);
      advanceOffset(sizeof(decltype(ggml_data_type)));

      auto tensor_offset =
          reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(tensor_offset)));

      ggml_tensor* t = nullptr;
      switch (dimension_count) {
      case 1:
        t = ggml_new_tensor_1d(weight_context, ggml_data_type,
                               tensor_dimensions[0]);
        break;
      case 2:
        t = ggml_new_tensor_2d(weight_context, ggml_data_type,
                               tensor_dimensions[0], tensor_dimensions[1]);
        break;
      case 3:
        t = ggml_new_tensor_3d(weight_context, ggml_data_type,
                               tensor_dimensions[0], tensor_dimensions[1],
                               tensor_dimensions[2]);
        break;
      case 4:
        t = ggml_new_tensor_4d(weight_context, ggml_data_type,
                               tensor_dimensions[0], tensor_dimensions[1],
                               tensor_dimensions[2], tensor_dimensions[3]);
        break;
      default:
        ERROR_AND_EXIT("Unsupported dimension count ", dimension_count);
      }

      if (tensor_name.find("token_embd.weight") != std::string::npos) {
        global_tensor.token_embd_weights = t;
      } else if (tensor_name.find("output_norm.weight") != std::string::npos) {
        global_tensor.output_norm_weights = t;
      } else if (tensor_name.find("output.weight") != std::string::npos) {
        global_tensor.output_weights = t;
      } else {
        auto layer_index = LayerIndex(tensor_name);
        ERRORIF(layer_index > model_block_count,
                "Layer Index Greater than the actualy block count ",
                layer_index);
        auto& block = blocks[layer_index];
        MapTensorToBlock(tensor_name, t, block);
      }

      uint32_t byte_size = 1;
      for (uint8_t i = 0; i < dimension_count; ++i) {
        byte_size *= tensor_dimensions[i];
      }

      const auto block_size = ggml_blck_size(ggml_data_type);
      ERRORIF(byte_size % block_size != 0, "Number of elements in tensor ",
              tensor_name, " is not a multiple of block size ", block_size);
      byte_size = byte_size * ggml_type_size(ggml_data_type) / block_size;

#ifdef GGUF_DEBUG
      std::cout << "\nTensor {\n";

      std::cout << "  Name   : " << tensor_name << "\n";
      std::cout << "  NDims  : " << dimension_count << "\n";

      std::cout << "  Shape  : [";
      for (size_t i = 0; i < dimension_count; ++i) {
        std::cout << tensor_dimensions[i];
        if (i != dimension_count - 1)
          std::cout << ", ";
      }
      std::cout << "]\n";
      std::cout << "  Offset : " << tensor_offset << "\n";
      std::cout << "  Size(in bytes) : " << byte_size << "\n";
      std::cout << "  Type   : " << ggml_type_name(ggml_data_type) << " ("
                << ggml_data_type << ")\n";
      std::cout << "}";
#endif
    }
    this->global_data_offset =
        (this->current_offset + this->byte_alignment - 1) &
        ~(this->byte_alignment - 1);
  }
  ~GGufReader() {
    cleanupResources();
  }
};
}; // namespace Odin
