#include "engine.hpp"
#include "forward.hpp"
#include "logging.hpp"

#include <cmath>
#include <span>

void Engine::Init(Model& model , ggml_gallocr* prefill_allocr , ggml_gallocr* infer_allocr, ggml_backend* target_backend , ggml_tensor* kcache , ggml_tensor* vcache , ggml_backend_buffer* backend_buffer){

  model_ = model;
  backend_ = target_backend;
  prefillAllocr_ = prefill_allocr;
  inferAllocr_ = infer_allocr;
  cache.Init(kcache , vcache , backend_buffer);

  state_.innerDimension = model.globals.embeddingLength / model.globals.attentionHeadCount;
  state_.scaleFactor = 1.0f / std::sqrt(state_.innerDimension);
  state_.pastTokenCount       = 0;

}

void Engine::ReserveDecodeMemory() {
  size_t s = 1;

  struct ggml_init_params params = {context_arena_size, NULL, true};
  ggml_context*           ctx0   = ggml_init(params);
  ggml_cgraph*            gf     = ggml_new_graph(ctx0);

  ggml_tensor* pos     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);
  ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);
  ggml_tensor* embeddings = ggml_get_rows(ctx0, model_.globalTensors.tokenEmbdWeights, indices);

  size_t original_n_past = state_.pastTokenCount;
  state_.pastTokenCount           = model_.globals.contextLength - 1;

  embeddings = forward(ctx0, gf, embeddings, pos, s, model_, cache, state_);

  ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);
  ggml_build_forward_expand(gf, max_idx);

  if (!ggml_gallocr_reserve(inferAllocr_, gf)) {
    Log("Failed to reserve memory for infer_allocr");
    exit(1);
  }

  state_.pastTokenCount = original_n_past;
  ggml_free(ctx0);
}

void Engine::ReservePrefillMemory() {
  struct ggml_init_params params = {context_arena_size, NULL, true};
  ggml_context*           ctx0   = ggml_init(params);
  ggml_cgraph*            gf     = ggml_new_graph(ctx0);

  ggml_tensor* pos =
    ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, prefill_batch_size);
  ggml_tensor* indices =
    ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, prefill_batch_size);

  ggml_tensor* embeddings =
    ggml_get_rows(ctx0, model_.globalTensors.tokenEmbdWeights, indices);
  embeddings = forward(ctx0, gf, embeddings, pos, prefill_batch_size, model_,
                       cache, state_);

  ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);

  ggml_build_forward_expand(gf, max_idx);

  if (!ggml_gallocr_reserve(prefillAllocr_, gf)) {
    Log("Failed to reserve memory for prefill_allocr");
    exit(1);
  }

  ggml_free(ctx0);
}

uint32_t Engine::Prefill(std::span<uint32_t>& tokens) {
  struct ggml_init_params params = {context_arena_size, NULL, true};
  ggml_context*           ctx0   = ggml_init(params);
  ggml_cgraph*            gf     = ggml_new_graph(ctx0);

  size_t s = tokens.size();

  ggml_tensor* pos     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);
  ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, s);

  ggml_tensor* embeddings =
    ggml_get_rows(ctx0, model_.globalTensors.tokenEmbdWeights, indices);
  embeddings = forward(ctx0, gf, embeddings, pos, s, model_, cache, state_);

  ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);

  ggml_build_forward_expand(gf, max_idx);
  ggml_gallocr_alloc_graph(prefillAllocr_, gf);

  std::vector<int32_t> pos_data(s);
  for (size_t p = 0; p < s; p++) {
    pos_data[p] = p + state_.pastTokenCount;
  }

  ggml_backend_tensor_set(pos, pos_data.data(), 0, s * sizeof(int32_t));
  ggml_backend_tensor_set(indices, tokens.data(), 0, s * sizeof(int32_t));

  ggml_backend_graph_compute(backend_, gf);

  int32_t next_token;
  ggml_backend_tensor_get(max_idx, &next_token, 0, sizeof(int32_t));

  state_.pastTokenCount += s;

  ggml_free(ctx0);
  return next_token;
}

uint32_t Engine::Infer(uint32_t prev_token) {
  struct ggml_init_params params = {context_arena_size, NULL, true};
  ggml_context*           ctx0   = ggml_init(params);
  ggml_cgraph*            gf     = ggml_new_graph(ctx0);

  ggml_tensor* pos     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
  ggml_tensor* indices = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);

  ggml_tensor* embeddings = ggml_get_rows(ctx0, model_.globalTensors.tokenEmbdWeights, indices);

  embeddings = forward(ctx0, gf, embeddings, pos, 1, model_, cache, state_);

  ggml_tensor* max_idx = ggml_argmax(ctx0, embeddings);

  ggml_build_forward_expand(gf, max_idx);
  ggml_gallocr_alloc_graph(inferAllocr_, gf);

  int32_t current_pos = state_.pastTokenCount;
  ggml_backend_tensor_set(pos, &current_pos, 0, sizeof(int32_t));
  ggml_backend_tensor_set(indices, &prev_token, 0, sizeof(int32_t));

  ggml_backend_graph_compute(backend_, gf);

  int32_t next_token;
  ggml_backend_tensor_get(max_idx, &next_token, 0, sizeof(int32_t));

  state_.pastTokenCount += 1;
  ggml_free(ctx0);
  return next_token;
}

void Engine::ClearContext() {
  state_.pastTokenCount = 0;
}
