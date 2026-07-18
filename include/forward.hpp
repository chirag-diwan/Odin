#pragma  once

#include "./block.hpp"
#include "./types.hpp"
#include "../external/ggml/include/ggml.h"

ggml_tensor* forward(
    ggml_context* temp_ctx,
    ggml_cgraph* gf,
    ggml_tensor* embeddings ,
    ggml_tensor* pos ,
    size_t s,
    Model& model,
    KVCache& cache,
    EngineState& state
);


