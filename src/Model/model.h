#include "../GGUF/gguf.h"
#include <cstdint>
#include <ggml.h>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace Odin {
enum ModelKeyQueryValueArch { HQA, MHA, GQA };

class Model {
private:
  GGufReader& gguf_reader;

  struct ggml_context*     weight_context;
  struct ggml_context*     key_value_context;
  std::unique_ptr<uint8_t> key_value_buffer;
  struct ggml_tensor*      key_cache_tensor;
  struct ggml_tensor*      value_cache_tensor;

  uint64_t layer_count;
  uint64_t sequence_length;
  uint64_t key_value_head_count;
  uint64_t head_dimension;

  uint8_t current_token_id;

  std::vector<Odin::ModelBlock>& blocks;
  ModelGlobalTensors&            global_tensors;

  ModelKeyQueryValueArch KQVArch;

private:
  uint64_t calculateKeyValueCacheByteSize() {
    GgufValue metadata_value =
        gguf_reader.metadata_key_values.at("qwen2.block_count");
    layer_count =
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   metadata_value.data);

    uint32_t batch_size = 1;

    metadata_value = gguf_reader.metadata_key_values.at("qwen2.context_length");
    sequence_length =
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   metadata_value.data);

    metadata_value =
        gguf_reader.metadata_key_values.at("qwen2.attention.head_count_kv");
    key_value_head_count =
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   metadata_value.data);

    metadata_value =
        gguf_reader.metadata_key_values.at("qwen2.embedding_length");
    uint32_t embedding_length =
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   metadata_value.data);

    metadata_value =
        gguf_reader.metadata_key_values.at("qwen2.attention.head_count");
    uint32_t attention_head_count =
        std::visit(mix{[](auto&) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   metadata_value.data);
    head_dimension            = embedding_length / attention_head_count;
    uint8_t bytes_per_element = 1;

    return 2 * layer_count * batch_size * sequence_length *
           key_value_head_count * head_dimension * bytes_per_element;
  }

public:
  Model(GGufReader& parsed_file, std::vector<ModelBlock>& blocks,
        ModelGlobalTensors& global_tensors)
      : gguf_reader(parsed_file), blocks(blocks),
        global_tensors(global_tensors) {
    struct ggml_init_params weight_initialization_parameters = {
        .mem_size   = ggml_tensor_overhead() * gguf_reader.header.tensor_count +
                      1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true

    };

    weight_context = ggml_init(weight_initialization_parameters);
    ERRORIF(weight_context == NULL, "Error initializing ggml weight context");

    if (gguf_reader.head_count > gguf_reader.head_count_kv) {
      KQVArch = ModelKeyQueryValueArch::GQA;
    } else if (gguf_reader.head_count == gguf_reader.head_count_kv) {
      KQVArch = ModelKeyQueryValueArch::HQA;
    } else if (gguf_reader.head_count < gguf_reader.head_count_kv) {
      KQVArch = ModelKeyQueryValueArch::MHA;
    }
  }

  void initializeComputeAndCache() {
    size_t cache_byte_size = calculateKeyValueCacheByteSize();

    key_value_buffer = std::unique_ptr<uint8_t>(new uint8_t[cache_byte_size]);

    struct ggml_init_params cache_initialization_parameters = {
        .mem_size   = cache_byte_size,
        .mem_buffer = key_value_buffer.get(),
        .no_alloc   = false,
    };

    key_value_context = ggml_init(cache_initialization_parameters);

    key_cache_tensor =
        ggml_new_tensor_4d(key_value_context, GGML_TYPE_F32, head_dimension,
                           key_value_head_count, sequence_length, layer_count);
    value_cache_tensor =
        ggml_new_tensor_4d(key_value_context, GGML_TYPE_F32, head_dimension,
                           key_value_head_count, sequence_length, layer_count);

    size_t                   compute_byte_size = 256ull * 1024 * 1024;
    std::unique_ptr<uint8_t> compute_memory_buffer(
        new uint8_t[compute_byte_size]);

    struct ggml_init_params compute_initialization_parameters = {
        .mem_size   = compute_byte_size,
        .mem_buffer = compute_memory_buffer.get(),
        .no_alloc   = false};

    struct ggml_context* compute_context =
        ggml_init(compute_initialization_parameters);

    struct ggml_tensor* token_tensor =
        ggml_new_tensor_1d(compute_context, GGML_TYPE_I32, 1);

    static_cast<int32_t*>(token_tensor->data)[0] = current_token_id;
  }

  void Inference(uint32_t TokenID) {
    for (const auto& block : blocks) {
    }
  }

  ~Model() {
    ggml_free(weight_context);
    ggml_free(key_value_context);
  }
};
} // namespace Odin
