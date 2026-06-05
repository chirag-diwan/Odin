#pragma once
#include "block.hpp"
#include "forward.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "logging.hpp"
#include "span.hpp"
#include "types.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

class Engine{
  private:
    EngineState state;
    Model model;

    const ggml_backend_t backend ;
    const ggml_gallocr_t prefill_allocr ;
    const ggml_gallocr_t infer_allocr ;

    KVCache cache;


  public:
    static constexpr size_t prefill_batch_size = 512;
    static constexpr size_t context_arena_size = 10*1024*1024;


    Engine(Model& model ,ggml_context* state_ctx,  ggml_backend_t target_backend) :
      model(model) ,
      backend(target_backend),
      prefill_allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))),
      infer_allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))),
      cache(state_ctx, target_backend, model)
  {

    state.d = model.globals.embedding_length / model.globals.attention_head_count;
    state.scale_factor = 1.0f / sqrt(static_cast<float>(state.d));
    state.n_past = 0;
  }


    void ReserveDecodeMemory() {
      size_t s = 1;

      struct ggml_init_params params = { context_arena_size, NULL, true };
      ggml_context* ctx0 = ggml_init(params);
      ggml_cgraph* gf = ggml_new_graph(ctx0);

      ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);
      ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);
      ggml_tensor* embeddings = ggml_get_rows(ctx0, model.global_tensors.token_embd_weights, indices); 

      size_t original_n_past = state.n_past;
      state.n_past = model.globals.context_length - 1; 

      embeddings = forward(ctx0, gf, embeddings, pos,s, model, cache, state);

      ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);
      ggml_build_forward_expand(gf, max_idx);

      if (!ggml_gallocr_reserve(infer_allocr, gf)) {
        Log("Failed to reserve memory for infer_allocr");
        exit(1);
      }

      state.n_past = original_n_past;
      ggml_free(ctx0);
    }


    void ReservePrefillMemory(){
      struct ggml_init_params params = { context_arena_size, NULL, true };
      ggml_context* ctx0 = ggml_init(params);
      ggml_cgraph* gf = ggml_new_graph(ctx0);


      ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, prefill_batch_size);
      ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, prefill_batch_size);

      ggml_tensor* embeddings = ggml_get_rows(ctx0, model.global_tensors.token_embd_weights, indices); 
      embeddings = forward(ctx0, gf, embeddings, pos,prefill_batch_size, model, cache, state);


      ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);

      ggml_build_forward_expand(gf, max_idx);

      if (!ggml_gallocr_reserve(prefill_allocr, gf)) {
        Log("Failed to reserve memory for prefill_allocr");
        exit(1);
      }

      ggml_free(ctx0);
    }

    uint32_t Prefill(span<uint32_t>& tokens){
      struct ggml_init_params params = { context_arena_size, NULL, true };
      ggml_context* ctx0 = ggml_init(params);
      ggml_cgraph* gf = ggml_new_graph(ctx0);

      size_t s = tokens.size();

      ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);
      ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);

      ggml_tensor* embeddings = ggml_get_rows(ctx0, model.global_tensors.token_embd_weights, indices); 
      embeddings = forward(ctx0, gf, embeddings, pos,s, model, cache,state);


      ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);

      ggml_build_forward_expand(gf, max_idx);
      ggml_gallocr_alloc_graph(prefill_allocr, gf);

      std::vector<int32_t> pos_data(s);
      for (size_t p = 0; p < s; p++){ 
        pos_data[p] = p + state.n_past;
      }

      ggml_backend_tensor_set(pos, pos_data.data(), 0, s * sizeof(int32_t));
      ggml_backend_tensor_set(indices, tokens.data(), 0, s * sizeof(int32_t));

      ggml_backend_graph_compute(backend, gf);

      int32_t next_token;
      ggml_backend_tensor_get(max_idx, &next_token, 0, sizeof(int32_t));

      state.n_past += s;

      ggml_free(ctx0);
      return next_token;
    }

    uint32_t Infer(uint32_t prev_token) {
      struct ggml_init_params params = { context_arena_size, NULL, true };
      ggml_context* ctx0 = ggml_init(params);
      ggml_cgraph* gf = ggml_new_graph(ctx0);

      ggml_tensor* pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
      ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);

      ggml_tensor* embeddings = ggml_get_rows(ctx0, model.global_tensors.token_embd_weights, indices);

      embeddings = forward(ctx0, gf, embeddings, pos, 1, model, cache, state);

      ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);

      ggml_build_forward_expand(gf, max_idx);
      ggml_gallocr_alloc_graph(infer_allocr, gf);

      int32_t current_pos = state.n_past;
      ggml_backend_tensor_set(pos, &current_pos, 0, sizeof(int32_t));
      ggml_backend_tensor_set(indices, &prev_token, 0, sizeof(int32_t));

      ggml_backend_graph_compute(backend, gf);

      int32_t next_token;
      ggml_backend_tensor_get(max_idx, &next_token, 0, sizeof(int32_t));

      state.n_past += 1;
      ggml_free(ctx0);
      return next_token;
    }

    ~Engine(){
      ggml_gallocr_free(infer_allocr);
      ggml_gallocr_free(prefill_allocr);
    }
};
