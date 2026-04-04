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
  GGUF&         gguf_file;
  ggml_context* weight_context;

  std::unordered_map<std::string_view, struct ggml_tensor*> tensors;

  uint64_t KV_CacheSize() {
    gguf_value v = gguf_file.metadata_kv["qwen2.block_count"];
    uint64_t   block_count =
        std::visit(mix{[](auto& v) {
                         ERROR_AND_EXIT("Unknown type for block_count");
                         return static_cast<uint64_t>(-1);
                       },
                       [](uint64_t& v) {
                         return static_cast<uint64_t>(v);
                       },
                       [](uint32_t& v) {
                         return static_cast<uint64_t>(v);
                       }},
                   v.data);
  }

public:
  Model(GGUF& parsed_file) : gguf_file(parsed_file) {
    struct ggml_init_params init_param = {
        .mem_size   = ggml_tensor_overhead() * gguf_file.header.tensor_count +
                      1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = false

    };

    weight_context = ggml_init(init_param);
    ERRORIF(weight_context == NULL, "Error initializing ggml weight context");
    for (const auto& bp : parsed_file.tensor_data) {
      struct ggml_tensor* tensor = nullptr;

      if (bp.ndim == 1) {
        tensor = ggml_new_tensor_1d(weight_context, bp.type, bp.dim[0]);
      } else if (bp.ndim == 2) {
        tensor =
            ggml_new_tensor_2d(weight_context, bp.type, bp.dim[0], bp.dim[1]);
      } else if (bp.ndim == 3) {
        tensor = ggml_new_tensor_3d(weight_context, bp.type, bp.dim[0],
                                    bp.dim[1], bp.dim[2]);
      } else if (bp.ndim == 4) {
        tensor = ggml_new_tensor_4d(weight_context, bp.type, bp.dim[0],
                                    bp.dim[1], bp.dim[2], bp.dim[3]);
      } else {
        ERROR_AND_EXIT("Unsupported tensor dimensions");
      }

      tensor->data = (void*)(parsed_file.data + parsed_file.global_data_offset +
                             bp.offset);

#ifdef MODEL_DEBUG
      ggml_set_name(tensor, std::string(bp.name).c_str());
#endif

      tensors[bp.name] = tensor;
    }
  }

  ~Model() {
    ggml_free(weight_context);
  }
};
} // namespace Odin
