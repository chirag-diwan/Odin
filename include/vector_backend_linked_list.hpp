#pragma once

#include <algorithm>
#include <bits/floatn-common.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>


struct unit_t{
  uint32_t val;
  int32_t prev;
  int32_t next;
};

class VBlist{
  private:
    void expand(){
      capacity_ = 2*capacity_;
      std::unique_ptr<unit_t[]> temp = std::make_unique<unit_t[]>(capacity_);
      std::copy_n(data_.get() , size_ , temp.get());
      data_ = std::move(temp);

    }

    int32_t head_ = -1;
    int32_t tail_ = -1;

  public:
    size_t size_;
    size_t capacity_;
    std::unique_ptr<unit_t[]> data_;


    VBlist(): size_(0) , capacity_(256) , data_(std::make_unique<unit_t[]>(capacity_)){
    }

    size_t size() const {
      return size_;
    }

    void push(uint32_t val){
      if(__builtin_expect(size_ >= capacity_ , false)){
        expand();
      }

      data_[size_] = unit_t{
        .val = val,
          .prev = tail_,
          .next = -1,
      };

      if(__builtin_expect(tail_ != -1 , true)){
        data_[tail_].next = size_;
      }


      if(__builtin_expect(head_ == -1 , false)){
        head_ = size_;
      }

      tail_ = size_;
      size_++;
    }

    void erase(size_t i) {
      if (i >= size_) return;

      int32_t node = (int32_t)i;

      if (node == head_) head_ = data_[node].next;
      if (node == tail_) tail_ = data_[node].prev;

      if (data_[node].prev != -1)
        data_[data_[node].prev].next = data_[node].next;

      if (data_[node].next != -1)
        data_[data_[node].next].prev = data_[node].prev;

      data_[node].prev = -1;
      data_[node].next = -1;
    }


    void clear() {
      size_ = 0;
      head_ = -1;
      tail_ = -1;
    }

    void push_in(std::vector<uint32_t>& vec) {
      int32_t current = head_;

      while (current != -1) {
        vec.push_back(data_[current].val);
        current = data_[current].next;
      }
    }

};
