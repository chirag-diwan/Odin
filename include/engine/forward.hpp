#pragma  once

#include "../block.hpp"
#include "ggml.h"
#include "../types.hpp"

ggml_tensor* forward(
    ggml_context* temp_ctx,
    ggml_cgraph* gf,
    ggml_tensor* embeddings ,
    ggml_tensor* pos ,
    size_t s,
    Model& model,
    KVCache& cache,
    EngineState& state
    ){
  for(size_t i = 0 ; i < model.blocks.size() ; i++){
    auto& block = model.blocks[i];

    ggml_tensor* normed = ggml_rms_norm(temp_ctx, embeddings, model.globals.attention_layer_norm_rms_epsilon);
    normed = ggml_mul(temp_ctx, normed, block.attn_norm_w);

    ggml_tensor* K = ggml_mul_mat(temp_ctx, block.attn_k_w, normed);
    if(block.attn_k_b != nullptr){
      K = ggml_add_inplace(temp_ctx, K, block.attn_k_b); 
    }

    ggml_tensor* Q = ggml_mul_mat(temp_ctx, block.attn_q_w, normed);
    if(block.attn_q_b != nullptr){
      Q = ggml_add_inplace(temp_ctx, Q, block.attn_q_b);
    }

    ggml_tensor* V = ggml_mul_mat(temp_ctx, block.attn_v_w, normed); 
    if(block.attn_v_b != nullptr){
      V = ggml_add_inplace(temp_ctx, V, block.attn_v_b);
    }


    ggml_tensor* Q_3D = ggml_reshape_3d(temp_ctx, Q, state.d, model.globals.attention_head_count, s);
    ggml_tensor* K_3D = ggml_reshape_3d(temp_ctx, K, state.d, model.globals.attention_head_count_kv, s);
    ggml_tensor* V_3D = ggml_reshape_3d(temp_ctx, V, state.d, model.globals.attention_head_count_kv, s);

    if(model.globals.general_model_architecture == Architecture::LLAMA3){
      Q_3D = ggml_rope_ext(temp_ctx, Q_3D, pos, model.global_tensors.rope_freq_weights, state.d, GGML_ROPE_TYPE_NORMAL, model.globals.context_length, model.globals.rope_freq_base, 1.0f, 32.0f, 1.0f, 4.0f, 1.0f);
      K_3D = ggml_rope_ext(temp_ctx, K_3D, pos, model.global_tensors.rope_freq_weights, state.d, GGML_ROPE_TYPE_NORMAL, model.globals.context_length, model.globals.rope_freq_base, 1.0f, 32.0f, 1.0f, 4.0f, 1.0f);

    }else if(model.globals.general_model_architecture == Architecture::QWEN2){
      Q_3D = ggml_rope_ext(temp_ctx, Q_3D, pos, model.global_tensors.rope_freq_weights, state.d, GGML_ROPE_TYPE_NEOX, model.globals.context_length, model.globals.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
      K_3D = ggml_rope_ext(temp_ctx, K_3D, pos, model.global_tensors.rope_freq_weights, state.d, GGML_ROPE_TYPE_NEOX, model.globals.context_length, model.globals.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    //Q_3D = ggml_cont(temp_ctx, ggml_permute(temp_ctx, Q_3D, 0, 2, 1, 3));
    //K_3D = ggml_cont(temp_ctx, ggml_permute(temp_ctx, K_3D, 0, 2, 1, 3));
    //V_3D = ggml_cont(temp_ctx, ggml_permute(temp_ctx, V_3D, 0, 2, 1, 3));
    Q_3D = ggml_permute(temp_ctx, Q_3D, 0, 2, 1, 3); //dhs -> dsh
    K_3D = ggml_permute(temp_ctx, K_3D, 0, 2, 1, 3);
    V_3D = ggml_permute(temp_ctx, V_3D, 0, 2, 1, 3);

    cache.AppendToKeyCache(temp_ctx, gf, K_3D, state.n_past , i);
    cache.AppendToValueCache(temp_ctx, gf, V_3D, state.n_past , i);

    int64_t active_tokens = state.n_past + s;

    size_t layer_offset = i * cache.K->nb[3];
    ggml_tensor* K_view = ggml_view_3d(temp_ctx, cache.K, state.d, active_tokens, model.globals.attention_head_count_kv, cache.K->nb[1], cache.K->nb[2], layer_offset);
    ggml_tensor* V_view = ggml_view_3d(temp_ctx, cache.V, active_tokens, state.d, model.globals.attention_head_count_kv, cache.V->nb[1], cache.V->nb[2], layer_offset);

    auto qk_t = ggml_mul_mat(temp_ctx, K_view, Q_3D);//dch_kv , dsh -> cdh_kv , dsh -> sch
    qk_t = ggml_scale(temp_ctx, qk_t, state.scale_factor);
    qk_t = ggml_diag_mask_inf(temp_ctx, qk_t,  state.n_past);
    qk_t = ggml_soft_max(temp_ctx, qk_t);

    ggml_tensor* attention_out = ggml_mul_mat(temp_ctx, V_view, qk_t);

    ggml_tensor* attn_perm = ggml_permute(temp_ctx, attention_out, 0, 2,1,3);
    ggml_tensor* attn_cont = ggml_cont(temp_ctx, attn_perm);
    ggml_tensor* attn_flat = ggml_reshape_2d(temp_ctx, attn_cont, model.globals.embedding_length, s);

    ggml_tensor* attn_out = ggml_mul_mat(temp_ctx, block.attn_output_w, attn_flat);
    embeddings = ggml_add(temp_ctx, embeddings, attn_out); 

    ggml_tensor* ffn_in = ggml_rms_norm(temp_ctx, embeddings, model.globals.attention_layer_norm_rms_epsilon);
    ffn_in = ggml_mul(temp_ctx, ffn_in, block.ffn_norm_w);

    ggml_tensor* ffn_up = ggml_mul_mat(temp_ctx, block.ffn_up_w, ffn_in);
    ggml_tensor* ffn_gate = ggml_mul_mat(temp_ctx, block.ffn_gate_w, ffn_in);
    ggml_tensor* ffn_gate_swish = ggml_silu(temp_ctx, ffn_gate);
    ggml_tensor* ffn_gate_out = ggml_mul(temp_ctx, ffn_gate_swish, ffn_up);
    ggml_tensor* ffn_down = ggml_mul_mat(temp_ctx, block.ffn_down_w, ffn_gate_out);//Potential Error

    embeddings = ggml_add(temp_ctx, embeddings, ffn_down);
  }

  if(s > 1){
    embeddings = ggml_view_1d( temp_ctx, embeddings, model.globals.embedding_length, embeddings->nb[1] * (s - 1) );
  }

  embeddings = ggml_rms_norm_inplace(temp_ctx,embeddings,model.globals.attention_layer_norm_rms_epsilon);
  embeddings = ggml_mul(temp_ctx, embeddings,model.global_tensors.output_norm_weights);
  if(model.global_tensors.output_weights != nullptr){
    embeddings = ggml_mul_mat(temp_ctx, model.global_tensors.output_weights, embeddings);
  }else{
    embeddings = ggml_mul_mat(temp_ctx, model.global_tensors.token_embd_weights , embeddings);
  }

  return embeddings;
}


