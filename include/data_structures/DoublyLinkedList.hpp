#include <cassert>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <strings.h>
#include <type_traits>
#include "definations.hpp"

template<typename value_type , bool debug = false>
class DoublyLinkedList {
  private:
    struct node {
      struct emptyCond{};
      value_type val;
      int32_t next;
      int32_t prev;
      [[no_unique_address]] std::conditional_t<debug, bool, emptyCond> isOccup;

      node() : next(-1), prev(-1){
        if constexpr (debug) {
          isOccup = false;
        }
      }
    };

    struct iterator {
      using iterator_category = std::forward_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using pointer = node*;
      using reference = node&;

      std::unique_ptr<node[]>& nodes;
      uint32_t size;
      pointer ptr;

      iterator(uint32_t idx , std::unique_ptr<node[]>& nodes_in , uint32_t size) : nodes(nodes_in)  , size(size) ,  ptr(idx == -1 ? nullptr : nodes_in.get() + idx){}

      reference operator*() const {
        return *ptr; 
      }

      pointer operator->() const {
        return ptr;
      }

      iterator& operator++() {
        if (ptr->next == -1) {
          ptr = nullptr;
        } else {
          ptr = nodes.get() + ptr->next;
        }
        return *this; 
      }

      bool operator==(const iterator& other) const {
        return ptr == other.ptr; 
      }

      bool operator!=(const iterator& other) const {
        return ptr != other.ptr; 
      }
    };


    int32_t first_;
    int32_t last_;
    uint32_t size_;
    uint32_t capacity_;

    std::unique_ptr<node[]> nodes = std::make_unique<node[]>(capacity_);

    void expand() {
      capacity_ = capacity_ * 2;
      auto temp = std::make_unique<node[]>(capacity_);
      std::copy( nodes.get(), nodes.get() + size_, temp.get());
      nodes.swap(temp);
    }


  public:

    DoublyLinkedList() : first_(-1) , last_(-1) , size_(0) , capacity_(128) , nodes(std::make_unique<node[]>(capacity_)){};

    ODIN_INLINE void clear(){
      size_ = 0;
      first_ = -1;
      last_ = -1;
    }

    void compact(){
      uint32_t idx = 0;
      int32_t current = first_;
      while(current != -1){
        nodes[idx] = nodes[current];
        if constexpr (debug){
          nodes[current].isOccup = false;
        }
        if(nodes[current].prev != -1){
          nodes[nodes[current].prev].next = idx;
        }

        if(nodes[current].next != -1){
          nodes[nodes[current].next].prev = idx;
        }

        current = nodes[current].next;
        last_ = idx;
        idx++;
      }
    }


    ODIN_INLINE int32_t next(int32_t idx) const { 
      return nodes[idx].next;
    }

    ODIN_INLINE int32_t first() const { 
      return first_;
    }

    ODIN_INLINE value_type& front(){
      if constexpr(debug){
        assert(first_ != -1);
      }
      return nodes[first_].val;
    }

    ODIN_INLINE value_type& back(){ 
      if constexpr(debug){
        assert(last_ != -1);
      }
      return nodes[last_].val;
    }

    bool empty() const {
      return size_ == 0;
    }

    void push(value_type val) {
      if constexpr (debug) {
        if(size_ < capacity_ && last_ > size_){
          compact();
        }
      }

      if (size_ == capacity_) {
        expand();
      }

      int32_t index = last_ + 1;

      nodes[index].val = val;
      nodes[index].next = -1;
      if constexpr (debug) {
        nodes[index].isOccup = true;
      }

      if (first_ == -1) {
        nodes[index].prev = -1;
        first_ = index;
      } else {
        nodes[index].prev = last_;
        nodes[last_].next = index;
      }

      last_ = index;
      ++size_;
    }
  
    ODIN_INLINE void erase(int32_t idx) {
      if constexpr (debug) {
        assert(idx <= last_);
        nodes[idx].isOccup = false;
      }

      int32_t node_next = nodes[idx].next;
      int32_t node_prev = nodes[idx].prev;

      if (node_prev != -1) {
        nodes[node_prev].next = node_next;
      } else {
        first_ = node_next;
      }

      if (node_next != -1) {
        nodes[node_next].prev = node_prev;
      }else{
        last_ = node_prev;
      }

      --size_;
    }

    iterator begin(){
      return iterator(first_, nodes , size_);
    }

    iterator end(){
      return iterator(-1, nodes , size_);
    }

    size_t size() const{
      return size_;
    }

    value_type& operator[](int32_t idx) {
      return nodes[idx].val; 
    }

    void displayFull() const {
      for(size_t i = 0 ; i < capacity_ ; i++){
        if constexpr (debug) {
          if(nodes[i].isOccup){
            std::cout << nodes[i].val;
          }else{
            std::cout << 'X';
          }
        }else{
          std::cout << nodes[i].val;
        }
        if(i != capacity_ - 1){
          std::cout << " -> ";
        }
      }
      std::cout << '\n';
    };

    void display() const {
      int32_t current = first_;

      std::cout << "[";

      while (current != -1) {
        std::cout << nodes[current].val;

        current = nodes[current].next;

        if (current != -1) {
          std::cout << ",";
        }
      }

      std::cout << "]\n";
    }
};

