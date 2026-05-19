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

class Model{
  private:
    ModelGlobals globals;
    ModelGlobalTensors global_tensors;

    ggml_tensor* K_cache;
    ggml_tensor* V_cache;

    size_t n_past;

    void appendToKeyCache( ggml_context* state_ctx , ggml_cgraph* gf, ggml_tensor* tensor, int token_index , size_t layer_index){
      size_t offset = K_cache->nb[3]*layer_index + K_cache->nb[1]*token_index;

      ggml_tensor* K_cache_view = ggml_view_3d(state_ctx,K_cache,tensor->ne[0],  tensor->ne[1], tensor->ne[2] ,K_cache->nb[1], K_cache->nb[2], offset);
      ggml_tensor* copy_node = ggml_cpy(state_ctx, tensor, K_cache_view);
      ggml_build_forward_expand(gf, copy_node);
    }

    void appendToValueCache( ggml_context* state_ctx , ggml_cgraph* gf, ggml_tensor* tensor, int token_index , size_t layer_index){

      size_t offset = V_cache->nb[3] * layer_index
        + V_cache->nb[0] * token_index;

      ggml_tensor* t = ggml_transpose(state_ctx, tensor);
      ggml_tensor* V_cache_view =
        ggml_view_3d(state_ctx,
            V_cache,
            t->ne[0],
            t->ne[1],
            t->ne[2],
            V_cache->nb[1],
            V_cache->nb[2],
            offset);

      ggml_tensor* copy_node = ggml_cpy(state_ctx, t, V_cache_view);
      ggml_build_forward_expand(gf, copy_node);
    }

    ggml_tensor* forward(ggml_context* temp_ctx, ggml_cgraph* gf, ggml_tensor* embeddings , ggml_tensor* pos , size_t s , size_t d , float scale_factor){
      for(size_t i = 0 ; i < blocks.size() ; i++){
        auto& block = blocks[i];

        ggml_tensor* normed = ggml_rms_norm(temp_ctx, embeddings, globals.attention_layer_norm_rms_epsilon);
        normed = ggml_mul(temp_ctx, normed, block.attn_norm_w);

        ggml_tensor* K = ggml_mul_mat(temp_ctx, block.attn_k_w, normed);
        K = ggml_add_inplace(temp_ctx, K, block.attn_k_b); 
        ggml_tensor* Q = ggml_mul_mat(temp_ctx, block.attn_q_w, normed);
        Q = ggml_add_inplace(temp_ctx, Q, block.attn_q_b);
        ggml_tensor* V = ggml_mul_mat(temp_ctx, block.attn_v_w, normed);
        V = ggml_add_inplace(temp_ctx, V, block.attn_v_b);

        ggml_tensor* Q_3D = ggml_reshape_3d(temp_ctx, Q, d, globals.attention_head_count, s);
        ggml_tensor* K_3D = ggml_reshape_3d(temp_ctx, K, d, globals.attention_head_count_kv, s);
        ggml_tensor* V_3D = ggml_reshape_3d(temp_ctx, V, d, globals.attention_head_count_kv, s);

        Q_3D = ggml_rope_ext(temp_ctx, Q_3D, pos, nullptr, d, GGML_ROPE_TYPE_NEOX, globals.context_length, globals.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K_3D = ggml_rope_ext(temp_ctx, K_3D, pos, nullptr, d, GGML_ROPE_TYPE_NEOX, globals.context_length, globals.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Q_3D = ggml_cont(temp_ctx, ggml_permute(temp_ctx, Q_3D, 0, 2, 1, 3));
        K_3D = ggml_cont(temp_ctx, ggml_permute(temp_ctx, K_3D, 0, 2, 1, 3));
        V_3D = ggml_cont(temp_ctx, ggml_permute(temp_ctx, V_3D, 0, 2, 1, 3));

        appendToKeyCache(temp_ctx, gf, K_3D, n_past , i);
        appendToValueCache(temp_ctx, gf, V_3D, n_past , i);

        int64_t active_tokens = n_past + s;

        size_t layer_offset = i * K_cache->nb[3];
        ggml_tensor* K_view = ggml_view_3d(temp_ctx, K_cache, d, active_tokens, globals.attention_head_count_kv, K_cache->nb[1], K_cache->nb[2], layer_offset);
        ggml_tensor* V_view = ggml_view_3d(temp_ctx, V_cache, active_tokens, d, globals.attention_head_count_kv, V_cache->nb[1], V_cache->nb[2], layer_offset);

        K_view = ggml_repeat_4d(temp_ctx, K_view, K_view->ne[0] , K_view->ne[1] , globals.attention_head_count , V_view->ne[3]);
        V_view = ggml_repeat_4d(temp_ctx, V_view, V_view->ne[0] , V_view->ne[1] , globals.attention_head_count , V_view->ne[3]);

        auto qk_t = ggml_mul_mat(temp_ctx, K_view, Q_3D);
        qk_t = ggml_scale(temp_ctx, qk_t, scale_factor);
        qk_t = ggml_diag_mask_inf(temp_ctx, qk_t,  n_past);
        qk_t = ggml_soft_max(temp_ctx, qk_t);

        ggml_tensor* attention_out = ggml_mul_mat(temp_ctx, V_view, qk_t);

        ggml_tensor* attn_perm = ggml_permute(temp_ctx, attention_out, 0, 2,1,3);
        ggml_tensor* attn_cont = ggml_cont(temp_ctx, attn_perm);
        ggml_tensor* attn_flat = ggml_reshape_2d(temp_ctx, attn_cont, globals.embedding_length, s);

        ggml_tensor* attn_out = ggml_mul_mat(temp_ctx, block.attn_output_w, attn_flat);
        embeddings = ggml_add(temp_ctx, embeddings, attn_out); 

        ggml_tensor* ffn_in = ggml_rms_norm(temp_ctx, embeddings, globals.attention_layer_norm_rms_epsilon);
        ffn_in = ggml_mul(temp_ctx, ffn_in, block.ffn_norm_w);

        ggml_tensor* ffn_up = ggml_mul_mat(temp_ctx, block.ffn_up_w, ffn_in);
        ggml_tensor* ffn_gate = ggml_mul_mat(temp_ctx, block.ffn_gate_w, ffn_in);
        ggml_tensor* ffn_gate_swish = ggml_silu(temp_ctx, ffn_gate);
        ggml_tensor* ffn_gate_out = ggml_mul(temp_ctx, ffn_gate_swish, ffn_up);
        ggml_tensor* ffn_down = ggml_mul_mat(temp_ctx, block.ffn_down_w, ffn_gate_out);

        embeddings = ggml_add(temp_ctx, embeddings, ffn_down);
      }

      if(s > 1){
        embeddings = ggml_view_1d( temp_ctx, embeddings, globals.embedding_length, embeddings->nb[1] * (s - 1) );
      }

      embeddings = ggml_rms_norm_inplace(temp_ctx,embeddings,globals.attention_layer_norm_rms_epsilon);
      embeddings = ggml_mul(temp_ctx, embeddings,global_tensors.output_norm_weights);
      embeddings = ggml_mul_mat(temp_ctx, global_tensors.output_weights, embeddings);

      // Return the terminal node. Do NOT execute or build_forward_expand here.
      return embeddings;
    }

  public:
    std::vector<ModelBlock> blocks;

    Model(MetadataKV_t& metadata_key_values){
      globals = {};
      global_tensors = {};
      K_cache = nullptr;
      V_cache = nullptr;
      n_past = 0;
      SetModelGlobals(metadata_key_values,globals);
    }


    void PopulateBlocks(std::vector<GGufTensor>& Tensors , ggml_context* tensor_context){
      blocks.resize(globals.block_count);
      for(const auto& tensor : Tensors){
        ggml_tensor* t;
        ggml_type current_type = tensor.tensor_type;
        switch (tensor.dimension_count){
          case 1:
            t = ggml_new_tensor_1d(tensor_context,current_type, tensor.dimensions[0]);
            break;
          case 2:
            t = ggml_new_tensor_2d(tensor_context,current_type, tensor.dimensions[0] , tensor.dimensions[1]);
            break;
          case 3:
            t = ggml_new_tensor_3d(tensor_context,current_type, tensor.dimensions[0] , tensor.dimensions[1] ,tensor.dimensions[2]);
            break;
          case 4:
            t = ggml_new_tensor_4d(tensor_context,current_type, tensor.dimensions[0] , tensor.dimensions[1] ,tensor.dimensions[2] , tensor.dimensions[3]);
            break;
          default:
            Log("Unknown dimension count " , tensor.dimension_count);
            continue;
        }

        t->data = tensor.weights_data;

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
      auto d_head = globals.embedding_length / globals.attention_head_count;

      auto c = globals.context_length; 
      auto n_head_kv = globals.attention_head_count_kv;

      K_cache = ggml_new_tensor_4d(state_ctx, GGML_TYPE_F16, 
          d_head, c, n_head_kv , globals.block_count);

      V_cache = ggml_new_tensor_4d(state_ctx, GGML_TYPE_F16, 
          c, d_head, n_head_kv , globals.block_count);
    }


    void Infer(std::vector<int>& tokens, int max_new_tokens = 40) {
      auto d = globals.embedding_length / globals.attention_head_count;
      float scale_factor = 1.0f / sqrt((float)d);

      ggml_backend_t backend = ggml_backend_cpu_init();
      ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

      for (int i = 0; i < max_new_tokens; i++) {
        struct ggml_init_params params = { 10 * 1024 * 1024, NULL, true };
        ggml_context* ctx0 = ggml_init(params);
        ggml_cgraph* gf = ggml_new_graph(ctx0);

        size_t s = (i == 0) ? tokens.size() : 1;

        ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);
        ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);

        ggml_tensor* embeddings = ggml_get_rows(ctx0, global_tensors.token_embd_weights, indices); 
        embeddings = forward(ctx0, gf, embeddings, pos, s, d, scale_factor);

        float temp_scale = 1.0f;
        ggml_tensor* scaled = ggml_scale(ctx0, embeddings, temp_scale);
        ggml_tensor* softmaxed = ggml_soft_max(ctx0, scaled);
        ggml_tensor* max_idx = ggml_argmax(ctx0, softmaxed);

        ggml_build_forward_expand(gf, max_idx);

        ggml_gallocr_alloc_graph(allocr, gf);

        if(i == 0){
          std::vector<int32_t> pos_data(s);
          for (size_t p = 0; p < s; p++) pos_data[p] = p;

          ggml_backend_tensor_set(pos, pos_data.data(), 0, s * sizeof(int32_t));
          ggml_backend_tensor_set(indices, tokens.data(), 0, s * sizeof(int32_t));
        } else {
          int32_t current_pos = n_past;
          ggml_backend_tensor_set(pos, &current_pos, 0, sizeof(int32_t));
          ggml_backend_tensor_set(indices, &tokens.back(), 0, sizeof(int32_t));
        }

        ggml_backend_graph_compute(backend, gf);

        int32_t next_token;
        ggml_backend_tensor_get(max_idx, &next_token, 0, sizeof(int32_t));

        tokens.push_back(next_token);
        n_past += s;

        ggml_free(ctx0);
      }

      ggml_gallocr_free(allocr);
      ggml_backend_free(backend);
    }
};
