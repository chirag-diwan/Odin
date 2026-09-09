#pragma once

#include "ggml.h"
#include <cstdint>
#include <string>
#include <vector>

//TODO In memory db , key as tensor and value as std::string
//TODO It should be a tree structure


class VectorDB{
  private:
    struct node{
      static constexpr size_t childrenCount = 16;
      ggml_tensor* key;
      std::string value;
      uint32_t children[childrenCount];
    };
    
    std::vector<node> nodePool_;
    
    ggml_tensor* getEmbedding(const std::string& key){
      //Tokenise this thing first
      //Then make a key embedding
    }

  public:
    VectorDB() = default;
    VectorDB(const VectorDB &) = default;
    VectorDB(VectorDB &&) = default;
    VectorDB &operator=(const VectorDB &) = default;
    VectorDB &operator=(VectorDB &&) = default;
  
    void Insert(const std::string& key , const std::string& value){
       
    }
};
