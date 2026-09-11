#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unistd.h>

template <typename T>
class Allocator{
  private:
    static constexpr size_t regionCount = 40;
    static constexpr size_t regionSize = 512;

    size_t capacity;
    size_t size_;

    void* space;


  public:
    using value_type = T;


    Allocator() : capacity(regionCount * regionSize) , size_(0){
      space = sbrk(capacity);
    }

    template<typename U> Allocator(const Allocator<U>& other) = delete;

    value_type* allocate(size_t count){
      if(size_ + count * sizeof(value_type) < capacity){
        auto temp = size_;
        size_ += count * sizeof(value_type);
        return reinterpret_cast<value_type*>(static_cast<uint8_t*>(space) + temp);
      }
      return nullptr;
    }

    size_t MaxPossibleValueCoubt(){
      return regionSize * regionCount * sizeof(value_type);
    }

    ~Allocator(){
      sbrk(-regionCount*regionSize);
    }
};
