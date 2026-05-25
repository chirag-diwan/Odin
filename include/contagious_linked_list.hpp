#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>


struct element_t{
  uint32_t val_;
  uint32_t next_;
  bool dead_;

  element_t(){
    val_ = 0;
    next_ = -1;
    dead_ = false;
  }

  element_t(uint32_t val , int16_t next , bool dead){
    val_ = val;
    next_ = next;
    dead_ = dead;
  }
};

class iterator {
  element_t* data_;
  int16_t index_;

  public:
  iterator(element_t* data, int16_t idx)
    : data_(data), index_(idx) {}

  element_t& operator*() {
    return data_[index_];
  }

  iterator& operator++() {
    index_ = data_[index_].next_;
    return *this;
  }

  bool operator!=(const iterator& other) const {
    return index_ != other.index_;
  }
};


template <size_t max_capacity>
class c_list {
  private:
    size_t size_;
    size_t used_;

    int32_t start_;
    int32_t tail_;

    std::unique_ptr<element_t[]> data_;

  public:
    c_list() {
      data_ = std::make_unique<element_t[]>(max_capacity);
      size_ = 0;
      used_ = 0;
      start_ = -1;
      tail_ = -1;
    }

    void clear(){
      size_ = 0;
      used_ = 0;
      start_ = -1;
      tail_ = -1;
      memset(data_.get(), 0, max_capacity*sizeof(element_t));
    }

    bool push(uint32_t val) {
      if (used_ >= max_capacity)
        return false;

      int16_t idx = used_++;

      data_[idx] = element_t(val, -1, false);

      if (tail_ != -1) {
        data_[tail_].next_ = idx;
      } else {
        start_ = idx;
      }

      tail_ = idx;
      ++size_;

      return true;
    }

    bool erase(int16_t idx) {
      if (idx < 0 || idx >= used_)
        return false;

      if (data_[idx].dead_)
        return true;

      int16_t prev = -1;
      int16_t cur = start_;

      while (cur != -1 && cur != idx) {
        prev = cur;
        cur = data_[cur].next_;
      }

      if (cur == -1)
        return false;

      if (prev == -1) {
        start_ = data_[cur].next_;
      } else {
        data_[prev].next_ = data_[cur].next_;
      }

      if (tail_ == idx)
        tail_ = prev;

      data_[idx].dead_ = true;

      --size_;

      return true;
    }

    iterator begin() {
      return iterator(data_.get(), start_);
    }

    iterator end() {
      return iterator(data_.get(), -1);
    }

    size_t size() const {
      return size_;
    }

    element_t* at(size_t i ){
      if(i < 0 || i >= size_){
        return nullptr;
      }
      if(!data_.get()[i].dead_){
        return &data_.get()[i];
      }
      return nullptr;
    }
};
