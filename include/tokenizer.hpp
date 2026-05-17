#include "errors.hpp"
#include "types.hpp"
#include <string_view>
#include <vector>
class Tokeniser{
  private:
    std::vector<std::string_view> tokens;

  public:
    Tokeniser(MetadataKV_t& metadata_kv){
      for(const auto & kv : metadata_kv){
        if (kv.name == "tokenizer.ggml.tokens") {
          Errorif(kv.value.type != GGUF_VALUE_TYPE_ARRAY, kv.name ,"should be a array type");
          Errorif(kv.value.array.elem_type != GGUF_VALUE_TYPE_STRING,"kv.value.array should be a string array");

          tokens = kv.value.array.strings;
        }
      }
    }

    void Tokenise(std::string prompt){
           
    }
};
