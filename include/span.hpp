#pragma once
#include <cstddef>
#include <vector>
template <typename T>
class span{
  private:
    T* data_;
    size_t size_;
    std::vector<T>& parent;
  public:
    span(std::vector<T>& vec , size_t offset , size_t size): parent(vec){
      data_ = vec.data() + offset;
      size_ = size > vec.size() ? vec.size() : size;
    }

    size_t size() const {
      return size_;
    }

    T* data(){
      return data_;
    }

    void push_back(T elem){
      parent.push_back(elem);
      data_ = parent.data();
      size_++;
    }

    T* begin(){
      return data_;
    }

    T* end(){
      return data_ + size_;
    }
};
