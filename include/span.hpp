#pragma once
#include <cstddef>
#include <cstdint>

template <typename value_type = uint32_t>
class span{
  private:
    const value_type* data_;
    size_t size_;

  public:
    span(value_type* data , size_t size) : data_(data) , size_(size){}

    size_t size() const {
      return size_;
    }

    const value_type* data() const {
      return data_;
    }

    value_type& operator[](size_t i){
      return data_[i];
    }

    value_type operator[](size_t i)const{
      return data_[i];
    }
};
