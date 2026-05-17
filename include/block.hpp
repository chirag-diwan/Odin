#pragma once

#include "logging.hpp"
#include <string>
#include <string_view>


struct ModelBlock {
  // Exists in: PreNorm, PostNorm.
  // Bias exists if: NormalizationType == LayerNorm
  struct ggml_tensor* attn_norm_w ;
  struct ggml_tensor* attn_norm_b ;

  // Exists in: PreNorm, PostNorm. Does NOT exist in Parallel (reuses
  // attn_norm).
  struct ggml_tensor* ffn_norm_w ;
  struct ggml_tensor* ffn_norm_b ;

  // Biases exist if: BiasTopology == Biased OR AttentionOnly
  // Layout: Separated (Llama, Qwen)
  struct ggml_tensor* attn_q_w ;
  struct ggml_tensor* attn_q_b ;
  struct ggml_tensor* attn_k_w ;
  struct ggml_tensor* attn_k_b ;
  struct ggml_tensor* attn_v_w ;
  struct ggml_tensor* attn_v_b ;

  // Layout: PackedQKV (Falcon, Bloom)
  struct ggml_tensor* attn_qkv_w ;
  struct ggml_tensor* attn_qkv_b ;

  // Layout: PackedKV (Some MQA variants)
  // Uses attn_q_w above, plus these combined KV weights
  struct ggml_tensor* attn_kv_w ;
  struct ggml_tensor* attn_kv_b ;

  // Always exists. Bias depends on BiasTopology.
  struct ggml_tensor* attn_output_w ;
  struct ggml_tensor* attn_output_b ;

  // Topology: Standard (GPT-2, Bloom)
  // ffn_up expands the dimension, ffn_down reduces it back.
  // Bias depends on BiasTopology.
  struct ggml_tensor* ffn_up_w   ;
  struct ggml_tensor* ffn_up_b   ;
  struct ggml_tensor* ffn_down_w ;
  struct ggml_tensor* ffn_down_b ;

  // Topology: Gated / SwiGLU (Llama, Qwen, Mistral)
  // Gate is the activation branch, Up is the linear branch. Down is the output.
  // Modern architectures rarely use biases here, but the engine must support
  // the possibility.
  struct ggml_tensor* ffn_gate_w ;
  struct ggml_tensor* ffn_gate_b ;
  // Uses ffn_up_w and ffn_down_w from above.

  ModelBlock(){
    attn_norm_w = nullptr;
    attn_norm_b = nullptr;
    ffn_norm_w = nullptr;
    ffn_norm_b = nullptr;
    attn_q_w = nullptr;
    attn_q_b = nullptr;
    attn_k_w = nullptr;
    attn_k_b = nullptr;
    attn_v_w = nullptr;
    attn_v_b = nullptr;
    attn_qkv_w = nullptr;
    attn_qkv_b = nullptr;
    attn_kv_w = nullptr;
    attn_kv_b = nullptr;
    attn_output_w = nullptr;
    attn_output_b = nullptr;
    ffn_up_w   = nullptr;
    ffn_up_b   = nullptr;
    ffn_down_w = nullptr;
    ffn_down_b = nullptr;
    ffn_gate_w = nullptr;
    ffn_gate_b = nullptr;
  }

  void MapTensor(const std::string_view& tensor_name , ggml_tensor* tensor){
    if (tensor_name.find(".attn_norm.weight") != std::string::npos)
      attn_norm_w = tensor;
    else if (tensor_name.find(".ffn_norm.weight") != std::string::npos)
      ffn_norm_w = tensor;
    else if (tensor_name.find(".attn_q.weight") != std::string::npos)
      attn_q_w = tensor;
    else if (tensor_name.find(".attn_q.bias") != std::string::npos)
      attn_q_b = tensor;
    else if (tensor_name.find(".attn_k.weight") != std::string::npos)
      attn_k_w = tensor;
    else if (tensor_name.find(".attn_k.bias") != std::string::npos)
      attn_k_b = tensor;
    else if (tensor_name.find(".attn_v.weight") != std::string::npos)
      attn_v_w = tensor;
    else if (tensor_name.find(".attn_v.bias") != std::string::npos)
      attn_v_b = tensor;
    else if (tensor_name.find(".attn_qkv.weight") != std::string::npos)
      attn_qkv_w = tensor;
    else if (tensor_name.find(".attn_output.weight") != std::string::npos)
      attn_output_w = tensor;
    else if (tensor_name.find(".ffn_gate.weight") != std::string::npos)
      ffn_gate_w = tensor;
    else if (tensor_name.find(".ffn_up.weight") != std::string::npos)
      ffn_up_w = tensor;
    else if (tensor_name.find(".ffn_down.weight") != std::string::npos)
      ffn_down_w = tensor;
  }
};

static inline const char* tstate(struct ggml_tensor* t) {
  return (t != NULL) ? "OK" : "NULL";
}

static inline void print_tensor(const char* name, struct ggml_tensor* t) {
  if (t == nullptr) {
    return;
  }
  Log(INFO,name, tstate(t), (void*)t);
}

void DebugPrintBlock(ModelBlock& block) {

  print_tensor("attn_norm_w", block.attn_norm_w);
  print_tensor("attn_norm_b", block.attn_norm_b);
  print_tensor("ffn_norm_w",  block.ffn_norm_w);
  print_tensor("ffn_norm_b",  block.ffn_norm_b);
  print_tensor("attn_q_w", block.attn_q_w);
  print_tensor("attn_q_b", block.attn_q_b);
  print_tensor("attn_k_w", block.attn_k_w);
  print_tensor("attn_k_b", block.attn_k_b);
  print_tensor("attn_v_w", block.attn_v_w);
  print_tensor("attn_v_b", block.attn_v_b);
  print_tensor("attn_qkv_w", block.attn_qkv_w);
  print_tensor("attn_qkv_b", block.attn_qkv_b);
  print_tensor("attn_kv_w", block.attn_kv_w);
  print_tensor("attn_kv_b", block.attn_kv_b);
  print_tensor("attn_output_w", block.attn_output_w);
  print_tensor("attn_output_b", block.attn_output_b);
  print_tensor("ffn_up_w",   block.ffn_up_w);
  print_tensor("ffn_up_b",   block.ffn_up_b);
  print_tensor("ffn_down_w", block.ffn_down_w);
  print_tensor("ffn_down_b", block.ffn_down_b);
  print_tensor("ffn_gate_w", block.ffn_gate_w);
  print_tensor("ffn_gate_b", block.ffn_gate_b);
}
