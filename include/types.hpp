#pragma once
#include "./block.hpp"
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
  uint64_t tensorCount;
  uint64_t metadataKvCount;

  GGufHeader(){
    magic = 0;
    version = 0;
    tensorCount = 0;
    metadataKvCount = 0;
  }

  GGufHeader(const GGufHeader &) = default;
  GGufHeader(GGufHeader &&) = default;
  GGufHeader &operator=(const GGufHeader &) = default;
  GGufHeader &operator=(GGufHeader &&) = default;
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
  uint32_t elemType;
  uint8_t* data;
  uint64_t length;
  std::vector<std::string_view> strings;

  GGufArray(){
    elemType = GGUF_VALUE_TYPE_NULL;
    data = nullptr;
    length = 0;
    strings = {};
  }

  GGufArray(const GGufArray &) = delete;
  GGufArray(GGufArray &&) = default;
  GGufArray &operator=(const GGufArray &) = delete;
  GGufArray &operator=(GGufArray &&) = default;
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

  GGufValue(const GGufValue &) = delete;
  GGufValue(GGufValue &&) = default;
  GGufValue &operator=(const GGufValue &) = delete;
  GGufValue &operator=(GGufValue &&) = default;
};

struct metadata_key_value{
  std::string_view name;
  GGufValue value;
};

struct GGufTensor {
  std::string_view name;
  ggml_type        tensorType;
  uint32_t         dimensionCount;
  int64_t          dimensions[DIM_ARRAY_MAX_SIZE];
  uint64_t         fileOffset;
  uint64_t         byteSize;
  uint8_t*         weightsData;

  GGufTensor(){
    dimensionCount = 1;
    for(int i = 0 ; i < DIM_ARRAY_MAX_SIZE ; i++){
      dimensions[i] = 1;
    }
    fileOffset = 0;
    byteSize = 1;
    weightsData = 0;
  }

  GGufTensor(const GGufTensor &) = default;
  GGufTensor(GGufTensor &&) = default;
  GGufTensor &operator=(const GGufTensor &) = default;
  GGufTensor &operator=(GGufTensor &&) = default;
};

enum class Architecture : uint8_t{
  QWEN2,
  LLAMA3,
  UNKNOWN
};

struct ModelGlobals{
  Architecture generalModelArchitecture;
  std::string_view fullArchitectureName;
  uint32_t blockCount;
  uint32_t embeddingLength;
  uint32_t feedForwardLength;
  uint32_t attentionHeadCount;
  uint32_t attentionHeadCountKv;
  uint32_t contextLength ;
  uint32_t ggmlEosTokenId;
  uint32_t ggmlBosTokenId;
  double ropeFreqBase ;
  double attentionLayerNormRmsEpsilon ;

  std::string_view chat_template;

  ModelGlobals(){
    generalModelArchitecture = Architecture::UNKNOWN;
    blockCount = 0;
    embeddingLength = 0;
    feedForwardLength = 0;
    attentionHeadCount = 0;
    attentionHeadCountKv = 0;
    contextLength  = 0;
    ropeFreqBase  = 0;
    attentionLayerNormRmsEpsilon  = 0;
    ggmlEosTokenId = 0;
    ggmlBosTokenId = 0;
  }

  ModelGlobals(Architecture generalModelArchitecture,
               std::string_view fullArchitectureName, uint32_t blockCount,
               uint32_t embeddingLength, uint32_t feedForwardLength,
               uint32_t attentionHeadCount, uint32_t attentionHeadCountKv,
               uint32_t contextLength, uint32_t ggmlEosTokenId,
               uint32_t ggmlBosTokenId, double ropeFreqBase,
               double attentionLayerNormRmsEpsilon)
      : generalModelArchitecture(generalModelArchitecture),
        fullArchitectureName(fullArchitectureName), blockCount(blockCount),
        embeddingLength(embeddingLength), feedForwardLength(feedForwardLength),
        attentionHeadCount(attentionHeadCount),
        attentionHeadCountKv(attentionHeadCountKv),
        contextLength(contextLength), ggmlEosTokenId(ggmlEosTokenId),
        ggmlBosTokenId(ggmlBosTokenId), ropeFreqBase(ropeFreqBase),
        attentionLayerNormRmsEpsilon(attentionLayerNormRmsEpsilon) {}
  ModelGlobals(const ModelGlobals &) = default;
  ModelGlobals(ModelGlobals &&) = default;
  ModelGlobals &operator=(const ModelGlobals &) = default;
  ModelGlobals &operator=(ModelGlobals &&) = default;
};

struct GlobalTensors {
  struct ggml_tensor* tokenEmbdWeights  ;
  struct ggml_tensor* outputNormWeights ;
  struct ggml_tensor* outputWeights      ;
  struct ggml_tensor* ropeFreqWeights   ;

  GlobalTensors(): tokenEmbdWeights(nullptr), outputNormWeights(nullptr), outputWeights(nullptr), ropeFreqWeights(nullptr){}

  GlobalTensors(const GlobalTensors &) = default;
  GlobalTensors(GlobalTensors &&) = default;
  GlobalTensors &operator=(const GlobalTensors &) = default;
  GlobalTensors &operator=(GlobalTensors &&) = default;
};

struct merge_rank_result{
  uint32_t mergeRank;
  uint32_t mergeResult;
};

struct rank_index_pair{
  uint64_t rank;
  uint64_t index;

  bool operator>(const rank_index_pair& other) const {
    return rank > other.rank;
  }
};


struct Config{
  std::string ipcPath;
  std::string modelPath;
  std::string tokeniserJsonPath;
  std::string historyPath;

  uint32_t port;
  uint8_t threadCount;

  Config(){
    ipcPath = "/tmp/odin0000.socket";
    historyPath = "/tmp/odin-prompt-history.txt";
    port = 8080;
    threadCount = std::thread::hardware_concurrency();
    modelPath = "NOT PROVIDED";
    tokeniserJsonPath = "NOT PROVIDED";
  }

  Config(const Config &) = default;
  Config(Config &&) = default;
  Config &operator=(const Config &) = default;
  Config &operator=(Config &&) = default;
};

struct Model{
  ModelGlobals globals;
  GlobalTensors globalTensors;
  std::vector<ModelBlock> blocks;
};

struct EngineState{
  size_t innerDimension;
  float scaleFactor;
  size_t pastTokenCount;
};

struct KVCache{
  ggml_tensor* K;
  ggml_tensor* V;
  ggml_backend_buffer_t kvBuffer;

  void Init(ggml_tensor* kcache , ggml_tensor* vcache , ggml_backend_buffer* backend_buffer) {
    K = kcache;
    V = vcache;
    kvBuffer = backend_buffer;
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
