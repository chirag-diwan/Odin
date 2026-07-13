#pragma once
#include "./block.hpp"
#include "./errors.hpp"
#include "../external/ggml/include/ggml-alloc.h"
#include "../external/ggml/include/ggml-backend.h"
#include "../external/ggml/include/ggml.h"
#include <cstdint>
#include <netinet/in.h>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#define DIM_ARRAY_MAX_SIZE 8 //Future proof

struct GGufHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;

  GGufHeader(){
    magic = 0;
    version = 0;
    tensor_count = 0;
    metadata_kv_count = 0;
  }

};



enum GGufValueType {
  GGUF_VALUE_TYPE_UINT8   = 0,
  GGUF_VALUE_TYPE_INT8    = 1,
  GGUF_VALUE_TYPE_UINT16  = 2,
  GGUF_VALUE_TYPE_INT16   = 3,
  GGUF_VALUE_TYPE_UINT32  = 4,
  GGUF_VALUE_TYPE_INT32   = 5,
  GGUF_VALUE_TYPE_FLOAT32 = 6,
  GGUF_VALUE_TYPE_BOOL    = 7,
  GGUF_VALUE_TYPE_STRING  = 8,
  GGUF_VALUE_TYPE_ARRAY   = 9,
  GGUF_VALUE_TYPE_UINT64  = 10,
  GGUF_VALUE_TYPE_INT64   = 11,
  GGUF_VALUE_TYPE_FLOAT64 = 12,
  GGUF_VALUE_TYPE_NULL,
};

struct GGufArray{
  uint32_t elem_type;
  uint8_t* data;
  uint64_t length;
  std::vector<std::string_view> strings;

  GGufArray(){
    elem_type = GGUF_VALUE_TYPE_NULL;
    data = nullptr;
    length = 0;
    strings = {};
  }
};

struct GGufValue {
  uint8_t* data;
  std::string_view string;
  GGufArray array;
  uint32_t type;

  GGufValue(){
    data = nullptr;
    type = GGUF_VALUE_TYPE_NULL;
  }
};


struct metadata_key_value{
  std::string_view name;
  GGufValue value;
};

using metadatakv_t = std::vector<metadata_key_value>;


struct GGufTensor {
  std::string_view name;
  ggml_type        tensor_type;
  uint32_t         dimension_count;
  int64_t          dimensions[DIM_ARRAY_MAX_SIZE];
  uint64_t         file_offset;
  uint64_t         byte_size;
  uint8_t*         weights_data;

  GGufTensor(){
    dimension_count = 1;
    for(int i = 0 ; i < DIM_ARRAY_MAX_SIZE ; i++){
      dimensions[i] = 1;
    }
    file_offset = 0;
    byte_size = 1;
    weights_data = 0;
  }
};


enum class Architecture : uint8_t{
  QWEN2,
  LLAMA3,
  UNKNOWN
};

struct ModelGlobals{
  Architecture general_model_architecture;
  std::string_view full_architecture_name;
  uint32_t block_count;
  uint32_t embedding_length;
  uint32_t feed_forward_length;
  uint32_t attention_head_count;
  uint32_t attention_head_count_kv;
  uint32_t context_length ;
  uint32_t ggml_eos_token_id;
  uint32_t ggml_bos_token_id;
  double rope_freq_base ;
  double attention_layer_norm_rms_epsilon ;

  const std::vector<std::string_view>* token_vocab;
  const std::vector<std::string_view>* token_merges;


  ModelGlobals(){
    general_model_architecture = Architecture::UNKNOWN;
    block_count = 0;
    embedding_length = 0;
    feed_forward_length = 0;
    attention_head_count = 0;
    attention_head_count_kv = 0;
    context_length  = 0;
    rope_freq_base  = 0;
    attention_layer_norm_rms_epsilon  = 0;
    ggml_eos_token_id = 0;
    ggml_bos_token_id = 0;
    token_vocab = nullptr;
    token_merges = nullptr;
  }
};

struct GlobalTensors {
  struct ggml_tensor* token_embd_weights  ;
  struct ggml_tensor* output_norm_weights ;
  struct ggml_tensor* output_weights      ;
  struct ggml_tensor* rope_freq_weights   ;

  GlobalTensors(){
    token_embd_weights  = nullptr;
    output_norm_weights = nullptr;
    output_weights      = nullptr;
    rope_freq_weights   = nullptr;
  }
};


struct merge_rank_result{
  uint32_t merge_rank;
  uint32_t merge_result;
};

struct rank_index_pair{
  uint64_t rank;
  uint64_t index;

  bool operator>(const rank_index_pair& other) const {
    return rank > other.rank;
  }
};


struct Config{
  std::string ipc_path;
  std::string model_path;
  std::string tokeniser_json_path;
  std::string history_path;

  bool use_ipc;
  bool use_http;
  uint32_t port;
  uint8_t thread_count;

  Config(){
    ipc_path = "/tmp/odin0000.socket";
    history_path = "/tmp/odin-prompt-history.txt";
    use_ipc = false;
    use_http = false;
    port = 8080;
    thread_count = std::thread::hardware_concurrency();
    model_path = "NOT PROVIDED";
    tokeniser_json_path = "NOT PROVIDED";
  }
};

struct InferenceParams{
  float temp;
  uint32_t K;
};

struct Model{
  ModelGlobals globals;
  GlobalTensors global_tensors;
  std::vector<ModelBlock> blocks;
};

struct EngineState{
  size_t d;
  float scale_factor;
  size_t n_past;
  float temp_inv;
};

struct KVCache{
  ggml_tensor* K;
  ggml_tensor* V;
  ggml_backend_buffer_t kv_buffer;

  KVCache(ggml_context* state_ctx ,ggml_backend_t backend ,  Model& model ){
    auto d_head = model.globals.embedding_length / model.globals.attention_head_count;

    auto c = model.globals.context_length; 
    auto n_head_kv = model.globals.attention_head_count_kv;

    K = ggml_new_tensor_4d(state_ctx, GGML_TYPE_F16, 
        d_head, c, n_head_kv , model.globals.block_count);

    V = ggml_new_tensor_4d(state_ctx, GGML_TYPE_F16, 
        c, d_head, n_head_kv , model.globals.block_count);

    kv_buffer = ggml_backend_alloc_ctx_tensors(state_ctx, backend);
    Errorif(kv_buffer == nullptr, "Failed to allocate physical memory for KV cache");
  }

  void AppendToKeyCache( ggml_context* state_ctx , ggml_cgraph* gf, ggml_tensor* tensor, int token_index , size_t layer_index)const{

    size_t offset = K->nb[3]*layer_index + K->nb[1]*token_index;

    ggml_tensor* K_view = ggml_view_3d(state_ctx,K,tensor->ne[0],  tensor->ne[1], tensor->ne[2] ,K->nb[1], K->nb[2], offset);
    ggml_tensor* copy_node = ggml_cpy(state_ctx, tensor, K_view);
    ggml_build_forward_expand(gf, copy_node);
  }

  void AppendToValueCache( ggml_context* state_ctx , ggml_cgraph* gf, ggml_tensor* tensor, int token_index , size_t layer_index)const{

    size_t offset = V->nb[3] * layer_index
      + V->nb[0] * token_index;

    ggml_tensor* t = ggml_transpose(state_ctx, tensor);
    ggml_tensor* V_view =
      ggml_view_3d(state_ctx,
          V,
          t->ne[0],
          t->ne[1],
          t->ne[2],
          V->nb[1],
          V->nb[2],
          offset);

    ggml_tensor* copy_node = ggml_cpy(state_ctx, t, V_view);
    ggml_build_forward_expand(gf, copy_node);
  }


  ~KVCache(){
    if (kv_buffer != nullptr) {
      ggml_backend_buffer_free(kv_buffer);
    }
  }
};


struct TokeniserConfig{
  bool turnacation;
  bool padding;
  std::string_view normalizer;
};

struct PreTokeniser{
  std::string_view type;
  std::string_view regex;
  std::string_view behavior;
  bool invert;
};

enum class PageType : uint8_t {
  KEY,
  VALUE
};

struct Page{
  static constexpr size_t PAGE_SIZE = 32;
  ggml_tensor* data;
  size_t size;

  Page(): data(nullptr) , size(0){ }
  Page(ggml_tensor* data): data(data) , size(0){ }

  bool full(){
    return size >= PAGE_SIZE;
  }
};

