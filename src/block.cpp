#include "../include/block.hpp"

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
