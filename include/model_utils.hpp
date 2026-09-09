#pragma once

#include <cstdint>
#include <string_view>
#include "./types.hpp"
#include "./gguf.hpp"
#include "./ggufparser.hpp"


uint64_t calculateKeyValueCacheByteSize(ModelGlobals& global_struct) {
  auto    head_dimension    = global_struct.embeddingLength / global_struct.attentionHeadCount;
  uint8_t bytes_per_element = 1;
  uint32_t batch_size = 1;

  return global_struct.blockCount * batch_size * global_struct.contextLength * global_struct.attentionHeadCountKv * head_dimension * bytes_per_element;
}

ModelGlobals GetModelGlobals(const std::vector<metadata_key_value>& metadata_key_values ){
  ModelGlobals global_struct;
  GGufValue metadata_value ;

  for(const auto& kv : metadata_key_values){
    if(kv.name.find("chat_template") != std::string_view::npos){
      global_struct.chat_template = kv.value.string;

    }else if(kv.name.find("block_count") != std::string_view::npos){
      global_struct.blockCount = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("context_length") != std::string_view::npos){
      global_struct.contextLength = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("attention.head_count_kv") != std::string_view::npos){
      global_struct.attentionHeadCountKv = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("embedding_length") != std::string_view::npos){
      global_struct.embeddingLength = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if(kv.name.find("attention.head_count") != std::string_view::npos){
      global_struct. attentionHeadCount = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if( kv.name.find("feed_forward_length") != std::string_view::npos){
      global_struct.feedForwardLength = Extract<uint64_t, GGUF_VALUE_TYPE_UINT32, GGUF_VALUE_TYPE_UINT64>(kv.value);

    }else if( kv.name.find("rope.freq_base") != std::string_view::npos){
      global_struct.ropeFreqBase = Extract<float, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64>(kv.value);

    }else if( kv.name.find("attention.layer_norm_rms_epsilon") != std::string_view::npos){
      global_struct.attentionLayerNormRmsEpsilon = Extract<float, GGUF_VALUE_TYPE_FLOAT32, GGUF_VALUE_TYPE_FLOAT64>(kv.value);

    }else if(kv.name.find("tokenizer.ggml.eos_token_id") != std::string_view::npos){
      global_struct.ggmlEosTokenId = Extract<uint32_t , GGUF_VALUE_TYPE_INT32 , GGUF_VALUE_TYPE_UINT32>(kv.value);

    }else if(kv.name.find("tokenizer.ggml.bos_token_id") != std::string_view::npos){
      global_struct.ggmlBosTokenId = Extract<uint32_t , GGUF_VALUE_TYPE_INT32 , GGUF_VALUE_TYPE_UINT32>(kv.value);

    }else if(kv.name.find("tokenizer.ggml.tokens") != std::string_view::npos){
      //global_struct.token_vocab = kv.value.array.strings.data();
      //global_struct.token_vocab_size = kv.value.array.strings.size();

    }else if(kv.name.find("tokenizer.ggml.merges") != std::string_view::npos){
      //global_struct.token_merges = kv.value.array.strings.data();
      //global_struct.token_merges_size = kv.value.array.strings.size();

    }else if(kv.name.find("general.architecture") != std::string_view::npos){
      global_struct.fullArchitectureName = kv.value.string;
      if(kv.value.string == "llama"){
        global_struct.generalModelArchitecture = Architecture::LLAMA3;
      }else if(kv.value.string == "qwen2"){
        global_struct.generalModelArchitecture = Architecture::QWEN2;
      }
    }
  }
  return global_struct;
}



Model CreateModel(ggml_context* tensor_context, const GGufParser& reader){
  Model m;
  m.globals = GetModelGlobals(reader.metadata_key_values_);
  if(m.globals.generalModelArchitecture == Architecture::UNKNOWN){
    //TODO Try and get more information about the Architecture using the full name.
    Log(ERROR , "Unknown model architecture" , m.globals.fullArchitectureName);
  }

  m.blocks.resize(m.globals.blockCount);

  for(const auto& tensor : reader.tensors_){
    ggml_tensor* t;
    ggml_type current_type = tensor.tensorType;
    switch (tensor.dimensionCount){
      case 1:
        t = ggml_new_tensor_1d(tensor_context,current_type, tensor.dimensions[0]);
        break;
      case 2:
        t = ggml_new_tensor_2d(tensor_context,current_type, tensor.dimensions[0] , tensor.dimensions[1]);
        break;
      case 3:
        t = ggml_new_tensor_3d(tensor_context,current_type, tensor.dimensions[0] , tensor.dimensions[1] ,tensor.dimensions[2]);
        break;
      case 4:
        t = ggml_new_tensor_4d(tensor_context,current_type, tensor.dimensions[0] , tensor.dimensions[1] ,tensor.dimensions[2] , tensor.dimensions[3]);
        break;
      default:
        Log("Unknown dimension count " , tensor.dimensionCount);
        continue;
    }

    t->data = tensor.weightsData;

    if(tensor.name == "token_embd.weight"){
      m.globalTensors.tokenEmbdWeights = t;

    }else if(tensor.name == "output.weight"){
      m.globalTensors.outputWeights = t;

    }else if(tensor.name == "output_norm.weight"){
      m.globalTensors.outputNormWeights = t;

    }else if(tensor.name == "rope_freqs.weight"){
      m.globalTensors.ropeFreqWeights = t;

    }else{
      auto layer_idx = LayerIndex(tensor.name);
      Errorif(layer_idx >= m.globals.blockCount, "Layer index greater than block count" , layer_idx , m.globals.blockCount);
      m.blocks[layer_idx].MapTensor(tensor.name, t);
    }
  }
  return m;
}
