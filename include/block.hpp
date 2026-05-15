#pragma once

#include <string>
#include <string_view>


struct ModelBlock {
  // Exists in: PreNorm, PostNorm.
  // Bias exists if: NormalizationType == LayerNorm
  struct ggml_tensor* attn_norm_w = nullptr;
  struct ggml_tensor* attn_norm_b = nullptr;

  // Exists in: PreNorm, PostNorm. Does NOT exist in Parallel (reuses
  // attn_norm).
  struct ggml_tensor* ffn_norm_w = nullptr;
  struct ggml_tensor* ffn_norm_b = nullptr;

  // Biases exist if: BiasTopology == Biased OR AttentionOnly
  // Layout: Separated (Llama, Qwen)
  struct ggml_tensor* attn_q_w = nullptr;
  struct ggml_tensor* attn_q_b = nullptr;
  struct ggml_tensor* attn_k_w = nullptr;
  struct ggml_tensor* attn_k_b = nullptr;
  struct ggml_tensor* attn_v_w = nullptr;
  struct ggml_tensor* attn_v_b = nullptr;

  // Layout: PackedQKV (Falcon, Bloom)
  struct ggml_tensor* attn_qkv_w = nullptr;
  struct ggml_tensor* attn_qkv_b = nullptr;

  // Layout: PackedKV (Some MQA variants)
  // Uses attn_q_w above, plus these combined KV weights
  struct ggml_tensor* attn_kv_w = nullptr;
  struct ggml_tensor* attn_kv_b = nullptr;

  // Always exists. Bias depends on BiasTopology.
  struct ggml_tensor* attn_output_w = nullptr;
  struct ggml_tensor* attn_output_b = nullptr;

  // Topology: Standard (GPT-2, Bloom)
  // ffn_up expands the dimension, ffn_down reduces it back.
  // Bias depends on BiasTopology.
  struct ggml_tensor* ffn_up_w   = nullptr;
  struct ggml_tensor* ffn_up_b   = nullptr;
  struct ggml_tensor* ffn_down_w = nullptr;
  struct ggml_tensor* ffn_down_b = nullptr;

  // Topology: Gated / SwiGLU (Llama, Qwen, Mistral)
  // Gate is the activation branch, Up is the linear branch. Down is the output.
  // Modern architectures rarely use biases here, but the engine must support
  // the possibility.
  struct ggml_tensor* ffn_gate_w = nullptr;
  struct ggml_tensor* ffn_gate_b = nullptr;
  // Uses ffn_up_w and ffn_down_w from above.

  ModelBlock() = default;
  
  void MapTensor(std::string_view& tensor_name , ggml_tensor* tensor){
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
