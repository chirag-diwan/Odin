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
#include <vector>

class Model{
  private:
    ModelGlobals globals;
    ModelGlobalTensors global_tensors;

    ggml_tensor* K_cache;
    ggml_tensor* V_cache;


    size_t curr_pos;

    void AppendToKVCache( ggml_context* ctx, ggml_tensor* cache, ggml_tensor* tensor, int token_index) {
      const int d = cache->ne[0];
      const int kv_heads = cache->ne[2];

      const int offset = token_index*K_cache->nb[1];

      ggml_tensor* cache_view = ggml_view_3d(ctx, cache, d, tensor->ne[1], kv_heads, cache->nb[1], cache->nb[2], offset);
      ggml_cpy(ctx, tensor, cache_view);
    }

  public:
    std::vector<ModelBlock> blocks;

    Model(MetadataKV_t& metadata_key_values){
      globals = {};
      global_tensors = {};
      K_cache = nullptr;
      V_cache = nullptr;
      curr_pos = 0;

      SetModelGlobals(metadata_key_values,globals);
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


      K_cache = ggml_new_tensor_3d(state_ctx, GGML_TYPE_F32, 
          d, max_ctx, kv_heads);

      V_cache = ggml_new_tensor_3d(state_ctx, GGML_TYPE_F32, 
          d, max_ctx, kv_heads);
    }

    /*
       [INFO]ffn_norm_w OK 0x7fdc2b511b00
       [INFO]attn_output_w OK 0x7fdc2b521da0
       [INFO]ffn_up_w OK 0x7fdc2b2bb190
       [INFO]ffn_down_w OK 0x7fdc2ae0deb0
       [INFO]ffn_gate_w OK 0x7fdc2b064820
       */


    void Prefill(std::vector<uint16_t>& tokens , ggml_context* compute_ctx){
      Errorif(K_cache == nullptr || V_cache == nullptr, "Cache not populated");

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

      for(const auto& block : blocks){
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

        AppendToKVCache(compute_ctx, K_cache, K_3D, curr_pos);
        AppendToKVCache(compute_ctx, V_cache, V_3D, curr_pos);

        int64_t active_tokens = curr_pos + s;
        ggml_tensor* K_view = ggml_view_3d(compute_ctx, K_cache, d, active_tokens, globals.attention_head_count_kv, K_cache->nb[1], K_cache->nb[2], 0);
        ggml_tensor* V_view = ggml_view_3d(compute_ctx, V_cache, d, active_tokens, globals.attention_head_count_kv, V_cache->nb[1], V_cache->nb[2], 0);

        ggml_tensor* attention_out = ggml_flash_attn_ext(compute_ctx, Q_3D, K_view, V_view, nullptr, scale_factor, 0.0f, 0.0f);

        attention_out = ggml_permute(compute_ctx, attention_out, 0, 2, 1, 3);
        attention_out = ggml_cont(compute_ctx, attention_out);
        attention_out = ggml_reshape_2d(compute_ctx, attention_out, globals.embedding_length, s);

        ggml_tensor* proj = ggml_mul_mat(compute_ctx, block.attn_output_w, attention_out);
        embeddings = ggml_add_inplace(compute_ctx, embeddings, proj);

//        ggml_tensor* ffn_expand = ggml_mul_mat(compute_ctx, block.ffn_up_w, embeddings);
//        ggml_swiglu(compute_ctx, ffn_expand);
//
//        ggml_tensor* ffn_gate = ggml_mul_mat(compute_ctx,block.ffn_gate_w, embeddings);
//        ggml_tensor* ffn_out = ggml_mul(compute_ctx,ffn_gate,ffn_expand);
//      
//        ggml_tensor* ffn_down = ggml_mul_mat(compute_ctx, block.ffn_down_w,ffn_out );
//        ggml_rms_norm_inplace(compute_ctx, ffn_down, 0);
//
//        ggml_mul_mat(compute_ctx, block.ffn_norm_w, ffn_down);
//        embeddings = ggml_add_inplace(compute_ctx, embeddings, ffn_down);
      }
      ggml_cgraph* gf = ggml_new_graph(compute_ctx);

      ggml_build_forward_expand(gf, embeddings);
      int n_threads = 4; // Set to your CPU core count for testing

      ggml_graph_compute_with_ctx(compute_ctx, gf, n_threads);
    }
};
