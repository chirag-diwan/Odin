#include "../GGUF/gguf.h"
#include <cmath>
#include <cstdint>
#include <cstring>
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
  uint64_t attention_head_count;

  std::vector<ModelBlock>& blocks;
  ModelGlobalTensors&      global_tensors;

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

    if (gguf_reader.model_head_count > gguf_reader.model_head_count_kv) {
      KQVArch = ModelKeyQueryValueArch::GQA;
    } else if (gguf_reader.model_head_count ==
               gguf_reader.model_head_count_kv) {
      KQVArch = ModelKeyQueryValueArch::HQA;
    } else if (gguf_reader.model_head_count < gguf_reader.model_head_count_kv) {
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
  }

  void Start(struct ggml_context*  compute_context,
             std::vector<int32_t>& Tokens) {

    // Initial prefill phase
    //  Get the embedding for the TOKENS from token_embd_w , if it has some sort
    //  of bias then add that
    auto         n_tokens = Tokens.size();
    ggml_tensor* input_tokens =
        ggml_new_tensor_1d(compute_context, GGML_TYPE_I32, Tokens.size());
    memcpy(input_tokens->data, Tokens.data(), n_tokens);

    // Tokens.clear(); // clear the tokens from the ram
    struct ggml_tensor* token_embeddings = ggml_get_rows(
        compute_context, global_tensors.token_embd_weights, input_tokens);
    // position tensor for prefill it has to be pupulated with all the position
    // i.e. from 0 to n_token -1 or from 1 to n_tokens , as you like
    ggml_tensor* pos =
        ggml_new_tensor_1d(compute_context, GGML_TYPE_I32, n_tokens);
    for (size_t i = 0; i < n_tokens; i++) {
      static_cast<int32_t*>(pos->data)[i] = i;
    }

    int     layer_index = 0;
    int32_t n_past = 0; // This should be passed into Start() as an argument,
                        // but assuming 0 for prefill.

    ggml_tensor* residual = token_embeddings;

    // Start the layer pass
    for (const auto& block : blocks) {
      //  Now get q , k , v . They come from the Wa Wk Wv matrix multiplied by
      //  token_embd
      ggml_tensor* Q =
          ggml_mul_mat(compute_context, block.attn_q_w, token_embeddings);
      // if bias is present then add it
      if (block.attn_q_b)
        ggml_add_inplace(compute_context, Q, block.attn_q_b);

      ggml_tensor* K =
          ggml_mul_mat(compute_context, block.attn_k_w, token_embeddings);
      if (block.attn_k_b)
        ggml_add_inplace(compute_context, K, block.attn_k_b);

      ggml_tensor* V =
          ggml_mul_mat(compute_context, block.attn_v_w, token_embeddings);
      if (block.attn_v_b)
        ggml_add_inplace(compute_context, V, block.attn_v_b);
      Q = ggml_reshape_3d(compute_context, Q, head_dimension,
                          attention_head_count, n_tokens);

      K = ggml_reshape_3d(compute_context, K, head_dimension,
                          key_value_head_count, n_tokens);
      ggml_rope_ext_inplace(compute_context, Q, pos, NULL, head_dimension, 0, 0,
                            gguf_reader.model_rope_freq_base, 1, 0, 0, 0, 0);
      ggml_rope_ext_inplace(compute_context, K, pos, NULL, head_dimension, 0, 0,
                            gguf_reader.model_rope_freq_base, 1, 0, 0, 0, 0);

      // Calculate the exact byte offset in the 4D cache for the current layer
      // and current position
      size_t cache_write_offset = (layer_index * key_cache_tensor->nb[3]) +
                                  (n_past * key_cache_tensor->nb[2]);

      // Create a view into the destination memory that matches the shape of K
      // and V exactly
      struct ggml_tensor* k_dst_view =
          ggml_view_3d(compute_context, key_cache_tensor, head_dimension,
                       key_value_head_count, n_tokens, key_cache_tensor->nb[1],
                       key_cache_tensor->nb[2], cache_write_offset);

      struct ggml_tensor* v_dst_view = ggml_view_3d(
          compute_context, value_cache_tensor, head_dimension,
          key_value_head_count, n_tokens, value_cache_tensor->nb[1],
          value_cache_tensor->nb[2], cache_write_offset);

      // copy the data from K into the destination
      // view ggml_cpy signature: ggml_cpy(ctx, src, dst)
      ggml_cpy(compute_context, K, k_dst_view);
      ggml_cpy(compute_context, V, v_dst_view);

      // Calculate the byte offset for the START of the current layer
      size_t layer_start_offset = layer_index * key_cache_tensor->nb[3];

      // The total context length we want to attend to
      int total_context_length = n_past + n_tokens;

      // Create views representing the history for this specific layer
      struct ggml_tensor* k_ctx_view = ggml_view_3d(
          compute_context, key_cache_tensor, head_dimension,
          key_value_head_count, total_context_length, key_cache_tensor->nb[1],
          key_cache_tensor->nb[2], layer_start_offset);

      struct ggml_tensor* v_ctx_view = ggml_view_3d(
          compute_context, value_cache_tensor, head_dimension,
          key_value_head_count, total_context_length, value_cache_tensor->nb[1],
          value_cache_tensor->nb[2], layer_start_offset);

      struct ggml_tensor* attn_out = ggml_flash_attn_ext(
          compute_context, Q, k_ctx_view, v_ctx_view,
          NULL, // No custom mask needed for standard generation
          1.0f / std::sqrtf(static_cast<float>(
                     head_dimension)), // Standard attention scaling
          0.0f,                        // max_bias
          0.0f                         // logit_softcap
      );

      struct ggml_tensor* cur = ggml_cont(
          compute_context, ggml_permute(compute_context, attn_out, 0, 2, 1, 3));
      cur = ggml_reshape_2d(compute_context, cur,
                            head_dimension * attention_head_count, n_tokens);

      // 2. Attention Output Projection
      cur = ggml_mul_mat(compute_context, block.attn_output_w, cur);
      if (block.attn_output_b) {
        ggml_add_inplace(compute_context, cur, block.attn_output_b);
      }

      // 3. First Residual Addition
      cur = ggml_add(compute_context, cur, residual);
      // Update residual for the next phase
      residual = cur;

      // 4. Post-Attention RMSNorm
      cur = ggml_rms_norm(compute_context, cur,
                          1e-6f); // Qwen2 norm eps is usually 1e-6
      cur = ggml_mul(compute_context, cur, block.ffn_norm_w);

      // 5. The SwiGLU Feed Forward Network
      // Route 1: Gate Projection + Activation
      struct ggml_tensor* ffn_gate =
          ggml_mul_mat(compute_context, block.ffn_gate_w, cur);
      ffn_gate = ggml_silu(compute_context, ffn_gate);

      // Route 2: Up Projection
      struct ggml_tensor* ffn_up =
          ggml_mul_mat(compute_context, block.ffn_up_w, cur);

      // Combine: Element-wise multiplication of Gate and Up
      cur = ggml_mul(compute_context, ffn_gate, ffn_up);

      // Down Projection
      cur = ggml_mul_mat(compute_context, block.ffn_down_w, cur);
      if (block.ffn_down_b) {
        ggml_add_inplace(compute_context, cur, block.ffn_down_b);
      }

      // 6. Second Residual Addition
      cur = ggml_add(compute_context, cur, residual);

      layer_index++;

      residual = cur;
    }
    // Now start with the actual token generation with the token that was
    // generated appended to the token vector
  }

  ~Model() {
    ggml_free(weight_context);
    ggml_free(key_value_context);
  }
};
} // namespace Odin
