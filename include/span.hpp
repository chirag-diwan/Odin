#pragma once
#include <cstddef>
#include <vector>
template <typename T>
class span{
  private:
    T* data_;
    size_t size_;
  public:
    span(std::vector<T>& vec , size_t offset , size_t size){
      data_ = vec.data() + offset;
      size_ = size > vec.size() ? vec.size() : size;
    }

    T* begin(){
      return data_;
    }

    T* end(){
      return data_ + size_;
    }
};
