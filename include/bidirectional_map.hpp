#pragma once
#include "logging.hpp"
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <sys/types.h>

template <typename key_type>
struct bid_key_t {
  key_type key;
  std::size_t value_index;
  bool occupied = false;
};

template <typename value_type>
struct bid_value_t {
  value_type value;
  std::size_t key_index;
  bool occupied = false;
};

template <typename key_type , typename value_type>
class bidirectional_map {
  private:
    std::unique_ptr<bid_key_t<key_type>[]> keys;
    std::unique_ptr<bid_value_t<value_type>[]> values;
    size_t capacity;
    size_t current_size;

    size_t getKeyIndex(key_type key) const {
      return std::hash<key_type>()(key) % capacity;
    }

    size_t getValueIndex(value_type value) const {
      return std::hash<value_type>()(value) % capacity;
    }

  public:
    explicit bidirectional_map() {
      capacity = 0;
      current_size = 0;
    }

    explicit bidirectional_map(size_t max_elements) {
      if (max_elements == 0) {
        return;
      }
      capacity = max_elements * 2;
      current_size = 0;

      keys = std::make_unique<bid_key_t<key_type>[]>(capacity);
      values = std::make_unique<bid_value_t<value_type>[]>(capacity);
    }

    void populate(size_t max_elements) {
      if (max_elements == 0) {
        return;
      }
      capacity = max_elements * 2;
      current_size = 0;

      keys = std::make_unique<bid_key_t<key_type>[]>(capacity);
      values = std::make_unique<bid_value_t<value_type>[]>(capacity);
    }

    bool insert(key_type key,value_type value) {
      if (current_size >= capacity / 2) {
        Log("Current size greator than capacity/2");
        std::exit(-1);
        return false; 
      }

      if(getKeyOf(value).has_value()) {
        Log("Value already present" , value);
        std::exit(-1);
        return false;
      }

      if (getValueOf(key).has_value()){
        Log("Key already present" , key);
        std::exit(-1);
        return false;
      }

      auto key_idx = getKeyIndex(key);
      while (keys[key_idx].occupied) {
        key_idx = (key_idx + 1) % capacity;
      }

      auto value_idx = getValueIndex(value);
      while (values[value_idx].occupied) {
        value_idx = (value_idx + 1) % capacity;
      }

      keys[key_idx] = bid_key_t<key_type>{.key = std::move(key), .value_index = value_idx, .occupied = true};
      values[value_idx] = bid_value_t<value_type>{.value = std::move(value), .key_index = key_idx, .occupied = true};
      current_size++;

      return true;
    }

    std::optional<key_type> getKeyOf(value_type value) const {
      auto value_idx = getValueIndex(value);
      while (values[value_idx].occupied) {
        if (values[value_idx].value == value) {
          return keys[values[value_idx].key_index].key;
        }
        value_idx = (value_idx + 1) % capacity;
      }
      return std::nullopt;
    }

    bool contains_key(key_type key) const{
      return getValueOf(key).has_value();
    }

    bool contains_value(value_type value) const{
      return getKeyOf(value).has_value();
    }

    std::optional<value_type> getValueOf(key_type key) const {
      auto key_idx = getKeyIndex(key);
      while (keys[key_idx].occupied) {
        if (keys[key_idx].key == key) {
          return values[keys[key_idx].value_index].value;
        }
        key_idx = (key_idx + 1) % capacity;
      }
      return std::nullopt;
    }

    size_t size() const {
      return current_size;
    }

    size_t max_size() const {
      return capacity;
    }
};
