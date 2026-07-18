#pragma  once
#include <functional>
#include <memory>
#include <optional>

template <typename key_type , typename obj_value_type>
class unidirectional_map{
  private:
    template <typename value_t , typename key_t>
      struct uni_pack_t{
        key_t key;
        value_t value;
        bool occupied;
      };

    class iterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = uni_pack_t<obj_value_type, key_type>;
        using difference_type   = std::ptrdiff_t;
        using pointer           = uni_pack_t<obj_value_type, key_type>*;
        using reference         = uni_pack_t<obj_value_type, key_type>&;

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

        friend bool operator==(const iterator& a, const iterator& b) { 
          return a.m_ptr == b.m_ptr; 
        }

        friend bool operator!=(const iterator& a, const iterator& b) { 
          return a.m_ptr != b.m_ptr; 
        }

      private:
        pointer m_ptr;
    };

    std::unique_ptr<uni_pack_t<obj_value_type, key_type>[]> values;
    size_t capacity;
    size_t current_size;

    size_t getIndexOf(key_type key){
      return std::hash<key_type>()(key)%capacity;
    }

  public:
    unidirectional_map(){
      capacity = 0;
      current_size = 0;
    }

    unidirectional_map(size_t max_size){
      if(max_size == 0){
        return;
      }
      capacity = 2*max_size;
      current_size = 0;
      values = std::make_unique<uni_pack_t<obj_value_type , key_type>[]>(capacity);
    }

    void populate(size_t max_size){
      if(max_size == 0){
        return;
      }
      capacity = 2*max_size;
      current_size = 0;
      values = std::make_unique<uni_pack_t<obj_value_type , key_type>[]>(capacity);
    }

    bool insert(key_type key , obj_value_type value){
      if (current_size >= capacity/2) {
        return false;
      }

      auto index = getIndexOf(key);
      while(values[index].occupied){
        index = (index + 1)%capacity;
      }

      values[index] = uni_pack_t<obj_value_type, key_type>{ .key = key, .value = value, .occupied = true };

      current_size ++;

      return true;
    }

    std::optional<obj_value_type> getValueOf(key_type key){
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
