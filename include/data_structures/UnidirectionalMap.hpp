#pragma  once
#include <functional>
#include <memory>
#include <optional>
#include "../logging.hpp"

template <typename key_type , typename val_type>
class UnidirectionalMap{
  private:
    struct uni_pack_t{
      key_type key;
      val_type value;
      bool occupied;
    };

    class iterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = uni_pack_t;
        using difference_type   = std::ptrdiff_t;
        using pointer           = uni_pack_t*;
        using reference         = uni_pack_t&;

        iterator(pointer ptr) : m_ptr(ptr) {}

        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }

        iterator& operator++() { 
          m_ptr++;
          return *this; 
        }  

        iterator operator++(int) { 
          iterator tmp = *this; 
          ++(*this); 
          return tmp; 
        }

        bool operator==(const iterator& b) { 
          return this->m_ptr == b.m_ptr; 
        }

        bool operator!=(const iterator& b) { 
          return this->m_ptr != b.m_ptr; 
        }

      private:
        pointer m_ptr;
    };

    std::unique_ptr<uni_pack_t[]> values;
    size_t capacity;
    size_t current_size;

    size_t getIndexOf(key_type key){
      return std::hash<key_type>()(key)%capacity;
    }

  public:
    UnidirectionalMap(){
      capacity = 0;
      current_size = 0;
    }

    UnidirectionalMap(size_t max_size){
      if(max_size == 0){
        return;
      }
      capacity = 2*max_size;
      current_size = 0;
      values = std::make_unique<uni_pack_t[]>(capacity);
    }

    void populate(size_t max_size){
      if(max_size == 0){
        return;
      }
      capacity = 2*max_size;
      current_size = 0;
      values = std::make_unique<uni_pack_t[]>(capacity);
    }

    [[nodiscard]]
      bool insert(key_type key , val_type value){
        if (current_size >= capacity/2) {
          Log(ERROR ,"Current size greator than capacity/2 (" , current_size , capacity , ")");
          return false;
        }

        auto index = getIndexOf(key);
        while(values[index].occupied){
          index = (index + 1)%capacity;
        }

        values[index] = uni_pack_t{ .key = key, .value = value, .occupied = true };

        current_size ++;

        return true;
      }

    ODIN_INLINE std::optional<val_type> getValueOf(key_type key){
      auto key_idx = getIndexOf(key);
      while (values[key_idx].occupied) {
        if (values[key_idx].key == key) {
          return values.get()[key_idx].value;
        }
        key_idx = (key_idx + 1) % capacity;
      }
      return std::nullopt;
    }

    iterator begin(){
      return iterator(values.get());
    }

    iterator end(){
      return iterator(values.get() + capacity);
    }
};
