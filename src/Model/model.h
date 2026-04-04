#include "../GGUF/gguf.h"
#include <cstdint>
#include <ggml.h>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace Odin {
class Model {
private:
  GGufReader&   gguf_file;
  ggml_context* weight_context;

  std::unordered_map<std::string_view, struct ggml_tensor*> tensors;

  uint64_t KvCacheSize() {
    GgufValue v = gguf_file.metadata_key_values.at("qwen2.block_count");
    uint64_t  layer_count =
        std::visit(mix{[](auto& unhandled_type) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   v.data);

    uint32_t batch_size = 1;

    v = gguf_file.metadata_key_values.at("qwen2.context_length");
    uint32_t sequence_length =
        std::visit(mix{[](auto& unhandled_type) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   v.data);

    v = gguf_file.metadata_key_values.at("qwen2.attention.head_count_kv");
    uint32_t kv_heads =
        std::visit(mix{[](auto& unhandled_type) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   v.data);

    v = gguf_file.metadata_key_values.at("qwen2.embedding_length");
    uint32_t embedding_length =
        std::visit(mix{[](auto& unhandled_type) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   v.data);

    v = gguf_file.metadata_key_values.at("qwen2.attention.head_count");
    uint32_t head_count =
        std::visit(mix{[](auto& unhandled_type) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& count) {
                         return static_cast<uint64_t>(count);
                       },
                       [](uint32_t& count) {
                         return static_cast<uint64_t>(count);
                       }},
                   v.data);
    uint32_t head_dimension = embedding_length / head_count;
    uint8_t  bytes          = 1;

    return layer_count * batch_size * sequence_length * kv_heads *
           head_dimension * bytes;
  }

public:
  Model(GGufReader& parsed_file) : gguf_file(parsed_file) {
    struct ggml_init_params init_param = {
        .mem_size   = ggml_tensor_overhead() * gguf_file.header.tensor_count +
                      1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false

    };

    weight_context = ggml_init(init_param);
    ERRORIF(weight_context == NULL, "Error initializing ggml weight context");
    for (const auto& bp : parsed_file.tensors) {
      struct ggml_tensor* tensor = nullptr;

      if (bp.dimension_count == 1) {
        tensor = ggml_new_tensor_1d(weight_context, bp.tensor_type,
                                    bp.dimensions[0]);
      } else if (bp.dimension_count == 2) {
        tensor = ggml_new_tensor_2d(weight_context, bp.tensor_type,
                                    bp.dimensions[0], bp.dimensions[1]);
      } else if (bp.dimension_count == 3) {
        tensor =
            ggml_new_tensor_3d(weight_context, bp.tensor_type, bp.dimensions[0],
                               bp.dimensions[1], bp.dimensions[2]);
      } else if (bp.dimension_count == 4) {
        tensor = ggml_new_tensor_4d(weight_context, bp.tensor_type,
                                    bp.dimensions[0], bp.dimensions[1],
                                    bp.dimensions[2], bp.dimensions[3]);
      } else {
        ERROR_AND_EXIT("Unsupported tensor dimensions");
      }

      tensor->data = (void*)(parsed_file.mapped_data +
                             parsed_file.global_data_offset + bp.file_offset);

#ifdef MODEL_DEBUG
      ggml_set_name(tensor, std::string(bp.name).c_str());
#endif

      tensors[bp.name] = tensor;
    }
  }

  void scratch() {
    size_t                  kv_size   = KvCacheSize();
    void*                   kv_buffer = new float[kv_size];
    struct ggml_init_params kv_params = {
        .mem_size = kv_size, .mem_buffer = kv_buffer, .no_alloc = false};
    ggml_context* ctx_kv = ggml_init(kv_params);
  }

  ~Model() {
    ggml_free(weight_context);
  }
};
} // namespace Odin
