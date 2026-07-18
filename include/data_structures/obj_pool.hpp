#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>
#include <type_traits>

template <typename obj_value_type, size_t maximum_size = 1024>
requires std::is_constructible_v<obj_value_type>
class object_pool{
  private:
    struct node_t{
      obj_value_type val;
      int32_t prev ;
      int32_t next ;
      bool occupied ;

      node_t() : prev(-1) , next(-1) , occupied(false){ }
      node_t(obj_value_type val , int64_t prev , int64_t next) : val(val) , prev(prev) , next(next) , occupied(true) { }
    };
  
    std::unique_ptr<node_t[]> data_;
    std::vector<uint32_t> free_list_;

    uint32_t size_;

    int32_t first_index_; 
    int32_t prev_index_ ;

  public:
    object_pool() : data_(std::make_unique<node_t[]>(maximum_size)) , free_list_(maximum_size) , size_(0) , first_index_(-1) , prev_index_(-1){
      std::iota(free_list_.rbegin(), free_list_.rend(), 0);
    }

    [[nodiscard]]
      std::optional<uint32_t> add(const obj_value_type& val){
        if(free_list_.empty()) return std::nullopt;

        auto free_idx = free_list_.back();free_list_.pop_back();

        if(first_index_ == -1)first_index_ = free_idx;

        data_[free_idx] = node_t(val , prev_index_ , -1);

        if(prev_index_ != -1) data_[prev_index_].next = free_idx;

        size_++;

        prev_index_ = free_idx;

        return free_idx;
      }

    [[nodiscard]]
      bool remove(size_t idx){
        if(size_ == 0)return false;
        if(idx >= maximum_size)return false;
        if(!data_[idx].occupied)return false;

        auto prev = data_[idx].prev;
        auto next = data_[idx].next;

        data_[idx].prev = -1;
        data_[idx].next = -1;
        data_[idx].occupied = false;

        if (prev != -1){
          data_[prev].next = next;
        } else{
          first_index_ = next;
        }

        if (next != -1){
          data_[next].prev = prev;
        } else{
          prev_index_ = prev;
        }

        free_list_.emplace_back(idx);

        size_--;

        return true;
      }

    const std::optional<node_t> at(size_t idx){
      if(idx >= maximum_size || (!data_[idx].occupied)){
        return std::nullopt;
      }
      return data_[idx];
    }

    const std::optional<node_t> first(){
      if(first_index_ == -1){
        return std::nullopt;
      }

      return data_[first_index_];
    }

    size_t size() const {
      return size_;
    }
};
