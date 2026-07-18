#include <cstdio>
#include "../include/pagemanager.hpp"
#include "../include/paged_attention.hpp"
#include "../external/ggml/include/ggml.h"
#include "../external/ggml/include/ggml-backend.h"

int main() {
  ggml_init_params params = {
    .mem_size   = 128 * 1024 * 1024, // 128 MB arena
    .mem_buffer = nullptr,
    .no_alloc   = false,
  };

  ggml_context * ctx = ggml_init(params);

  if (ctx == nullptr) {
    fprintf(stderr, "Failed to initialize ggml context\n");
    return 1;
  }

  printf("ggml context initialized\n");

  ggml_backend_t backend = ggml_backend_init_by_name("CPU", nullptr);

  if (backend == nullptr) {
    fprintf(stderr, "Failed to initialize ggml backend\n");
    ggml_free(ctx);
    return 1;
  }

  printf("ggml backend initialized\n");
  
  Model m;

  m.globals.embedding_length = 896;
  m.globals.attention_head_count = 14;
  m.globals.attention_head_count_kv = 2;
  m.globals.context_length = 32768;
  m.globals.block_count = 24;

  PageManager manager(ctx , backend , m);
  Cache k;
  Cache v;

  auto scratch = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Page::PAGE_SIZE, m.globals.attention_head_count);

  auto q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, m.globals.embedding_length , 1, m.globals.attention_head_count);
  
  page_attention_user_data user_data{
    .kcache = k,
    .vcache = v,
    .scratch_pad = scratch,
    .scratch_pad2 = nullptr,
    .scratch_pad3 = nullptr,
    .layer_index = 0,
    .token_index = 0
  };
  
  auto out = ggml_map_custom1(ctx, q, odin_paged_attention, 4 , &user_data);


  ggml_cgraph* graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, out);

  //---------------------------------------------
  // Execute
  //---------------------------------------------
  ggml_backend_graph_compute(backend, graph);


  ggml_backend_free(backend);
  ggml_free(ctx);

  return 0;
}
