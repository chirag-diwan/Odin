#pragma once

#include "span.hpp"
#include <utility>
#include <stdexcept>

template <typename value_type>
class vector {
  private:
    value_type* data_;
    size_t size_;
    size_t capacity_;

    void expand() {
      capacity_ = capacity_ == 0 ? 1 : capacity_ * 2;
      auto temp = new value_type[capacity_];

      for (size_t i = 0; i < size_; ++i) {
        temp[i] = std::move(data_[i]);
      }
      delete[] data_;
      data_ = temp;
    }

  public:
    vector()
      : size_(0),
      capacity_(256),
      data_(new value_type[capacity_])
  {}

    // 1. Destructor prevents the memory leak
    ~vector() {
      delete[] data_;
    }

    // 2. Copy Constructor
    vector(const vector& other) 
      : size_(other.size_), capacity_(other.capacity_), data_(new value_type[other.capacity_]) {
        for (size_t i = 0; i < size_; ++i) {
          data_[i] = other.data_[i];
        }
      }

    // 3. Move Constructor
    vector(vector&& other) noexcept 
      : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
      }

    // 4. Copy Assignment
    vector& operator=(const vector& other) {
      if (this != &other) {
        delete[] data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        data_ = new value_type[capacity_];
        for (size_t i = 0; i < size_; ++i) {
          data_[i] = other.data_[i];
        }
      }
      return *this;
    }

    // 5. Move Assignment
    vector& operator=(vector&& other) noexcept {
      if (this != &other) {
        delete[] data_;
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;

        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
      }
      return *this;
    }

    size_t size() const {
      return size_;
    }

    void clear() {
      size_ = 0;
    }

    void push(const value_type& val) {
      if (size_ >= capacity_) {
        expand();
      }
      data_[size_] = val;
      ++size_;
    }

    span<value_type> view(size_t start, size_t size) {
      if (start + size > size_) {
        throw std::out_of_range("Span view exceeds vector size");
      }
      return span<value_type>(data_, start, size);
    }

    const value_type& back() const {
      if (size_ == 0) {
        throw std::out_of_range("Cannot call back() on empty vector");
      }
      return data_[size_ - 1]; // Corrected OOB access
    }

    value_type* data() {
      return data_;
    }
};
