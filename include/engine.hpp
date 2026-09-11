#pragma once
#include "./types.hpp"
#include "ggml/include/ggml-alloc.h"
#include "ggml/include/ggml-backend.h"
#include "ggml/include/ggml.h"
#include <cstdint>
#include <cstdlib>
#include <span>

class Engine {
private:
  EngineState state_;
  Model       model_;

  ggml_backend_t backend_;
  ggml_gallocr_t prefillAllocr_;
  ggml_gallocr_t inferAllocr_;

  KVCache cache;

public:
  static constexpr size_t prefill_batch_size = 512;
  static constexpr size_t context_arena_size = 10 * 1024 * 1024;

  void Init(Model& model , ggml_gallocr* prefill_allocr , ggml_gallocr* infer_allocr, ggml_backend_t target_backend , ggml_tensor* kcache , ggml_tensor* vcache , ggml_backend_buffer* backend_buffer);

  void ReserveDecodeMemory() ;

  void ReservePrefillMemory() ;

  uint32_t Prefill(std::span<uint32_t>& tokens) ;

  uint32_t Infer(uint32_t prev_token) ;

  void ClearContext() ;
};
