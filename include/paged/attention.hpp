#pragma once

#include "ggml.h"

struct page_data{
  size_t block_count;
};

void page_attention(struct ggml_tensor * dst , const struct ggml_tensor * a, const struct ggml_tensor * b, const struct ggml_tensor * c, int ith, int nth, void * userdata){ 
  //GGML_ASSERT();
  //ggml_vec_dot_f16(int n, float *__restrict s, NULL, ggml_fp16_t *__restrict x, NULL, ggml_fp16_t *__restrict y, NULL, NULL);
}
