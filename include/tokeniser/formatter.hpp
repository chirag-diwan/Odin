#pragma once

#include <string>
#include <format>
#include "../errors.hpp"
#include "../types.hpp"

class Formatter{
  private:
  public:
    Formatter(){ }

    static std::string GetFormatted(Architecture model_arch , const std::string& system , const std::string & user){
      switch (model_arch) {
        case Architecture::LLAMA3:
          return std::format("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" , system , user);
        case Architecture::QWEN2:
          return std::format( "<|im_begin|>{}<|im_end|><|im_begin|> User , {} <|im_begin|>Assistant " , system , user);
        default:
          Errorif(true, "Invalid model architecture");
          return "";
      }
    }
};
