#pragma once

#include <atomic>
#include <memory>
#include <optional>


template<typename T , size_t init_capacity = 1024>
class ringbuffer{
  private:
    static constexpr size_t mask = init_capacity - 1;
  public:
    size_t capacity_;
    std::unique_ptr<T[]> data_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;

    explicit ringbuffer() :
      capacity_(init_capacity),
      data_(std::make_unique<T[]>(capacity_)),
      head_(0),
      tail_(0) {
        static_assert(!(init_capacity & (init_capacity - 1)), "init capacity must be power of two");
      }

    [[nodiscard]]
      bool push(const T& val){
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_acquire);

        if((tail - head) == capacity_){
          return false;
        }

        data_[tail&mask] = val;
        tail_.store(tail + 1 , std::memory_order_release);
        return true;
      }

    [[nodiscard]]
      std::optional<T> pop(){
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_acquire);

        if(tail == head){
          return std::nullopt;
        }

        auto val = data_[head&mask];
        head_.store(head + 1 , std::memory_order_release);
        return val;
      }

    [[nodiscard]]
      bool empty(){
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_acquire);

        if(head == tail){
          return true;
        }

        return false;
      }


    [[nodiscard]]
      size_t size(){
        auto tail = tail_.load(std::memory_order_relaxed);
        auto head = head_.load(std::memory_order_acquire);

        return tail - head;
      }


    [[nodiscard]]
      size_t capacity(){
        return capacity_ - size();
      }
  
    ringbuffer(ringbuffer& buffer) = delete;
    ringbuffer(ringbuffer&& buffer) = default;
};
