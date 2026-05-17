#pragma once
#include "block.hpp"
#include "errors.hpp"
#include "ggml-cpu.h"
#include "ggml.h"
#include "logging.hpp"
#include "model_utils.hpp"
#include "types.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

// Add this to your Model class: ggml_tensor* global_causal_mask;


class Model{
  private:
    ModelGlobals globals;
    ModelGlobalTensors global_tensors;

    ggml_tensor* K_cache;
    ggml_tensor* V_cache;
    ggml_tensor* global_causal_mask;

    size_t curr_pos;


    void AppendToKVCache( ggml_context* ctx, ggml_tensor* cache, ggml_tensor* tensor, int token_index , size_t layer_index) {
      const int64_t d         = cache->ne[0];
      const int64_t kv_heads  = cache->ne[2];

      size_t token_offset =
        token_index * cache->nb[1];

      size_t layer_offset =
        layer_index * cache->nb[3];

      size_t offset =
        token_offset + layer_offset;

      ggml_tensor* cache_view = ggml_view_3d( ctx, cache, d, tensor->ne[1], kv_heads, cache->nb[1], cache->nb[2], offset);

      tensor = ggml_cpy(ctx, tensor, cache_view);
    }

  public:
    std::vector<ModelBlock> blocks;

    Model(MetadataKV_t& metadata_key_values){
      globals = {};
      global_tensors = {};
      global_causal_mask = nullptr;
      K_cache = nullptr;
      V_cache = nullptr;
      curr_pos = 0;
      SetModelGlobals(metadata_key_values,globals);
    }

    void PopulateCausalMask(ggml_context* static_ctx) {
    }

    void PopulateBlocks(std::vector<GGufTensor>& Tensors , ggml_context* weight_context){
      blocks.resize(globals.block_count);
      for(const auto& tensor : Tensors){
        ggml_tensor* t;
        ggml_type current_type = tensor.tensor_type;
        switch (tensor.dimension_count){
          case 1:
            t = ggml_new_tensor_1d(weight_context,current_type, tensor.dimensions[0]);
            break;
          case 2:
            t = ggml_new_tensor_2d(weight_context,current_type, tensor.dimensions[0] , tensor.dimensions[1]);
            break;
          case 3:
            t = ggml_new_tensor_3d(weight_context,current_type, tensor.dimensions[0] , tensor.dimensions[1] ,tensor.dimensions[2]);
            break;
          case 4:
            t = ggml_new_tensor_4d(weight_context,current_type, tensor.dimensions[0] , tensor.dimensions[1] ,tensor.dimensions[2] , tensor.dimensions[3]);
            break;
          default:
            Log("Unknown dimension count " , tensor.dimension_count);
            continue;
        }

        if(tensor.name == "token_embd.weight" ){
          global_tensors.token_embd_weights = t;
        }else if(tensor.name == "output.weight"){
          global_tensors.output_weights = t;
        }else if(tensor.name == "output_norm.weight"){
          global_tensors.output_norm_weights = t;
        }else{
          auto layer_idx = LayerIndex(tensor.name);
          Errorif(layer_idx >= globals.block_count, "Layer index greater than block count" , layer_idx , globals.block_count);
          blocks[layer_idx].MapTensor(tensor.name, t);
        }
      }
    }

    void PopulateKVCache(ggml_context* state_ctx) {
      auto d = globals.embedding_length / globals.attention_head_count;

      auto max_ctx = globals.context_length; 
      auto kv_heads = globals.attention_head_count_kv;

      K_cache = ggml_new_tensor_4d(state_ctx, GGML_TYPE_F32, 
          d, max_ctx, kv_heads , globals.block_count);

      V_cache = ggml_new_tensor_4d(state_ctx, GGML_TYPE_F32, 
          d, max_ctx, kv_heads , globals.block_count);
    }

    ggml_tensor* forward(ggml_context* compute_ctx , ggml_tensor* embeddings , ggml_tensor* pos , size_t s , size_t d , float scale_factor){

      for(size_t i = 0 ; i < blocks.size() ; i++){
        auto& block = blocks[i];
        ggml_tensor* normed = ggml_rms_norm(compute_ctx, embeddings, globals.attention_layer_norm_rms_epsilon);
        normed = ggml_mul(compute_ctx, normed, block.attn_norm_w);

        ggml_tensor* K = ggml_mul_mat(compute_ctx, block.attn_k_w, normed);
        K = ggml_add_inplace(compute_ctx, K, block.attn_k_b); 
        ggml_tensor* Q = ggml_mul_mat(compute_ctx, block.attn_q_w, normed);
        Q = ggml_add_inplace(compute_ctx, Q, block.attn_q_b);
        ggml_tensor* V = ggml_mul_mat(compute_ctx, block.attn_v_w, normed);
        V = ggml_add_inplace(compute_ctx, V, block.attn_v_b);

        ggml_tensor* Q_3D = ggml_reshape_3d(compute_ctx, Q, d, globals.attention_head_count, s);
        ggml_tensor* K_3D = ggml_reshape_3d(compute_ctx, K, d, globals.attention_head_count_kv, s);
        ggml_tensor* V_3D = ggml_reshape_3d(compute_ctx, V, d, globals.attention_head_count_kv, s);

        Q_3D = ggml_rope_ext(compute_ctx, Q_3D, pos, nullptr, d, 0, globals.context_length, globals.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_3D = ggml_rope_ext(compute_ctx, K_3D, pos, nullptr, d, 0, globals.context_length, globals.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Q_3D = ggml_permute(compute_ctx, Q_3D, 0, 2, 1, 3);
        K_3D = ggml_permute(compute_ctx, K_3D, 0, 2, 1, 3);
        V_3D = ggml_permute(compute_ctx, V_3D, 0, 2, 1, 3);

        AppendToKVCache(compute_ctx, K_cache, K_3D, curr_pos , i);
        AppendToKVCache(compute_ctx, V_cache, V_3D, curr_pos , i);
        int64_t active_tokens = curr_pos + s;

        size_t layer_offset = i * K_cache->nb[3];
        ggml_tensor* K_view = ggml_view_3d(compute_ctx, K_cache, d, active_tokens, globals.attention_head_count_kv, K_cache->nb[1], K_cache->nb[2], layer_offset);
        ggml_tensor* V_view = ggml_view_3d(compute_ctx, V_cache, d, active_tokens, globals.attention_head_count_kv, V_cache->nb[1], V_cache->nb[2], layer_offset);

        ggml_tensor* mask = nullptr;
        if (s > 1) {
          mask = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F16, active_tokens, s);
          ggml_fp16_t mask_val = ggml_fp32_to_fp16(-INFINITY);
          ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
          ggml_fp16_t* mask_data = (ggml_fp16_t*)mask->data;
          int past_tokens = active_tokens - s;

          for (int r = 0; r < s; ++r) {
            for (int c = 0; c < active_tokens; ++c) {
              if (c > past_tokens + r) {
                mask_data[r * active_tokens + c] = mask_val; // Future token -> Block
              } else {
                mask_data[r * active_tokens + c] = zero_val; // Past/Current token -> Allow
              }
            }
          }
        }

        ggml_tensor* attention_out = ggml_flash_attn_ext(compute_ctx, Q_3D, K_view, V_view, mask, scale_factor, 0.0f, 0.0f);
        ggml_diag_mask_inf_inplace(compute_ctx, attention_out, curr_pos);

        attention_out = ggml_permute(compute_ctx, attention_out, 0, 2, 1, 3);
        attention_out = ggml_cont(compute_ctx, attention_out);
        attention_out = ggml_reshape_2d(compute_ctx, attention_out, globals.embedding_length, s);

        ggml_tensor* proj = ggml_mul_mat(compute_ctx, block.attn_output_w, attention_out);
        embeddings = ggml_add_inplace(compute_ctx, embeddings, proj);


        ggml_tensor* embed_norm = ggml_rms_norm(compute_ctx, embeddings, globals.attention_layer_norm_rms_epsilon);
        embed_norm = ggml_mul(compute_ctx,embed_norm , block.ffn_norm_w);

        ggml_tensor* ffn_expand = ggml_mul_mat(compute_ctx, block.ffn_up_w, embed_norm);

        ggml_tensor* ffn_gate = ggml_mul_mat(compute_ctx,block.ffn_gate_w, embed_norm);
        ffn_gate = ggml_swiglu(compute_ctx, ffn_gate);

        ggml_tensor* ffn_out = ggml_mul(compute_ctx,ffn_expand,ffn_gate);
        ggml_tensor* ffn_down = ggml_mul_mat(compute_ctx, block.ffn_down_w,ffn_out );
        embeddings = ggml_add_inplace(compute_ctx, embeddings, ffn_down);
      }
      curr_pos += s;
      return embeddings;
    }

    void Prefill(ggml_context* compute_ctx , std::vector<int>& tokens ){
      Errorif(K_cache == nullptr || V_cache == nullptr, "Cache not populated");
      Log(INFO , "Prefill start");

      auto d = globals.embedding_length / globals.attention_head_count;
      float scale_factor = 1.0f / sqrt((float)d);
      auto s = tokens.size();

      ggml_tensor* pos = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_I32, s);
      for (size_t i = 0; i < s; i++) {
        static_cast<int32_t*>(pos->data)[i] = i; 
      }

      ggml_tensor* indices = ggml_new_tensor_1d(compute_ctx,GGML_TYPE_I32, s);
      for(size_t i = 0 ; i < tokens.size() ; i++){
        static_cast<int32_t*>(indices->data)[i] = tokens[i];
      }

      ggml_tensor* embeddings = ggml_get_rows(compute_ctx,global_tensors.token_embd_weights, indices); // What i understood it should be seq_len x model_d(embedding_length)
      embeddings = forward(compute_ctx, embeddings,pos, s,d,  scale_factor);

      ggml_cgraph* gf = ggml_new_graph(compute_ctx);

      ggml_build_forward_expand(gf, embeddings);
      int n_threads = 6; // Set to your CPU core count for testing

      ggml_graph_compute_with_ctx(compute_ctx, gf, n_threads);

      Log(INFO , "Prefill completed successfully");
    }

    void Infer(std::vector<int>& tokens){
      auto d = globals.embedding_length / globals.attention_head_count;
      float scale_factor = 1.0f / sqrt((float)d);

      for (int i = 0; i < 40; i++) {
        struct ggml_init_params params = { 
          /* .mem_size   = */ 256 * 1024 * 1024, // Size depends on your model
          /* .mem_buffer = */ NULL,
          /* .no_alloc   = */ false 
        };
        ggml_context* temp_ctx = ggml_init(params);

        int s = 1;
        int current_token = tokens.back(); 

        ggml_tensor* pos = ggml_new_tensor_1d(temp_ctx, GGML_TYPE_I32, s);
        static_cast<int32_t*>(pos->data)[0] = curr_pos;

        ggml_tensor* indices = ggml_new_tensor_1d(temp_ctx, GGML_TYPE_I32, s);
        static_cast<int32_t*>(indices->data)[0] = current_token;

        // Embedding Lookup
        ggml_tensor* state = ggml_get_rows(temp_ctx, global_tensors.token_embd_weights, indices);

        state = forward(temp_ctx, state, pos, s, d, scale_factor);

        state = ggml_rms_norm(temp_ctx, state, globals.attention_layer_norm_rms_epsilon);
        state = ggml_mul(temp_ctx, state, global_tensors.output_norm_weights);
        ggml_tensor* logits = ggml_mul_mat(temp_ctx, global_tensors.output_weights, state);

        ggml_tensor* scaled_logits = ggml_scale(temp_ctx, logits, 1.0f / 0.4f);
        ggml_soft_max_inplace(temp_ctx, scaled_logits);
        ggml_tensor* token_tensor = ggml_argmax(temp_ctx, scaled_logits);

        ggml_cgraph* gf = ggml_new_graph(temp_ctx);
        ggml_build_forward_expand(gf, token_tensor);
        ggml_graph_compute_with_ctx(temp_ctx, gf, 4);

        int next_token = static_cast<int32_t*>(token_tensor->data)[0];
        tokens.push_back(next_token);

        ggml_free(temp_ctx); // Crucial: Free ephemeral memory

      }
    }
};
