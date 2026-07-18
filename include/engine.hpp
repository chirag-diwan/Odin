#pragma once
#include "./types.hpp"
#include "../external/ggml/include/ggml-alloc.h"
#include "../external/ggml/include/ggml-backend.h"
#include "../external/ggml/include/ggml.h"
#include <cstdint>
#include <cstdlib>
#include <span>

class Engine {
private:
  EngineState state;
  Model       model;

  const ggml_backend_t backend;
  const ggml_gallocr_t prefill_allocr;
  const ggml_gallocr_t infer_allocr;

  KVCache cache;

public:
  static constexpr size_t prefill_batch_size = 512;
  static constexpr size_t context_arena_size = 10 * 1024 * 1024;

  Engine(Model& model, ggml_context* state_ctx, ggml_backend_t target_backend);

  void ReserveDecodeMemory() ;

  void ReservePrefillMemory() ;

  uint32_t Prefill(std::span<uint32_t>& tokens) ;

  uint32_t Infer(uint32_t prev_token) ;

  void ClearContext() ;

  ~Engine() ;
};
