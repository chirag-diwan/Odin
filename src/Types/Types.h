#pragma once

#include <cstdint>
#include <vector>

extern "C" {
#include "../gguf-tools/gguflib.h"
}

#define QK8_0 32
#define QK5_0 32
#define QK_K 256

#define DIM_LENGHT 8
typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;
    int8_t    qs[QK8_0];
} block_q8_0;

typedef struct {
    ggml_half d;
    uint8_t   qh[4];
    uint8_t   qs[QK5_0 / 2];
} block_q5_0;

typedef struct {
    uint8_t   ql[QK_K / 2];
    uint8_t   qh[QK_K / 4];
    int8_t    scales[QK_K / 16];
    ggml_half d;
} block_q6_K;

typedef struct {
    union {
        struct {
            ggml_half d;
            ggml_half dmin;
        } GGML_COMMON_AGGR_S;
        uint16_t dm[2];
    } GGML_COMMON_AGGR_U;

    uint8_t scales[3 * QK_K / 64];
    uint8_t qs[QK_K / 2];
} block_q4_K;

struct Matrix {
    uint8_t*         data                  = nullptr;
    uint64_t         dimension[DIM_LENGHT] = {1, 1, 1, 1, 1, 1, 1, 1};
    gguf_tensor_type quant_type;
};

struct Layer {
    struct Matrix attn_norm;
    struct Matrix ffn_down;
    struct Matrix ffn_gate;
    struct Matrix ffn_up;
    struct Matrix ffn_norm;
    struct Matrix attn_k_bias;
    struct Matrix attn_k;
    struct Matrix attn_output;
    struct Matrix attn_q_bias;
    struct Matrix attn_q;
    struct Matrix attn_v_bias;
    struct Matrix attn_v;
};

struct Transformer {
    uint8_t*           output;
    uint8_t*           tokenEmbedding;
    std::vector<Layer> layers;

    Transformer() {
      layers = std::vector<Layer>(64);
    }
};
