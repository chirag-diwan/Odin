#pragma once

#include <sys/cdefs.h>
#include <sys/types.h>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include "../logging.hpp"

template <typename key_type, typename value_type>
class BidirectionalMap {
  private:
    struct bid_key_t {
      key_type key;
      std::size_t value_index;
      bool occupied = false;
    };

    struct bid_value_t {
      value_type value;
      std::size_t key_index;
      bool occupied = false;
    };

    std::unique_ptr<bid_key_t[]> keys;
    std::unique_ptr<bid_value_t[]> values;
    size_t capacity;
    size_t current_size;

    size_t getKeyIndex(const key_type& key) const {
      return std::hash<key_type>()(key) % capacity;
    }

    size_t getValueIndex(value_type value) const {
      return std::hash<value_type>()(value) % capacity;
    }

  public:
    class iterator {
      private:
        using key_ptr = bid_key_t*;
        using value_ptr = bid_value_t*;

        key_ptr keys;
        value_ptr values;
        size_t index;
        size_t capacity;

        void skip_invalid() {
          while (index < capacity && !keys[index].occupied) {
            ++index;
          }
        }

      public:
        iterator(key_ptr k, value_ptr v, size_t i, size_t cap)
          : keys(k), values(v), index(i), capacity(cap) {
            skip_invalid();
          }

        iterator& operator++() {
          ++index;
          skip_invalid();
          return *this;
        }

        bool operator!=(const iterator& other) const {
          return index != other.index;
        }

        std::pair<const key_type&, const value_type&> operator*() const {
          const auto& k = keys[index];
          const auto& v = values[k.value_index];
          return {k.key, v.value};
        }
    };

    BidirectionalMap() {
      capacity = 0;
      current_size = 0;
    }

    BidirectionalMap(size_t max_elements) {
      if (max_elements == 0) {
        return;
      }
      capacity = max_elements * 2;
      current_size = 0;

      keys = std::make_unique<bid_key_t[]>(capacity);
      values = std::make_unique<bid_value_t[]>(capacity);
    }

    void populate(size_t max_elements) {
      if (max_elements == 0) {
        return;
      }
      capacity = max_elements * 2;
      current_size = 0;

      keys = std::make_unique<bid_key_t[]>(capacity);
      values = std::make_unique<bid_value_t[]>(capacity);
    }

    [[nodiscard]]
      bool insert(const key_type& key, const value_type& value) {
        if (current_size >= capacity / 2) {
          Log(ERROR ,"Current size greator than capacity/2 (" , current_size , capacity , ")");
          return false;
        }

        if (getKeyOf(value).has_value()) {
          Log(ERROR ,"Value already present", value);
          return false;
        }

        if (getValueOf(key).has_value()) {
          Log(ERROR ,"Key already present", key);
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

        keys[key_idx] = bid_key_t{
          .key = std::move(key),
            .value_index = value_idx,
            .occupied = true
        };

        values[value_idx] = bid_value_t{
          .value = std::move(value),
            .key_index = key_idx,
            .occupied = true
        };
        current_size++;

        return true;
      }

    ODIN_INLINE std::optional<key_type> getKeyOf(const value_type& value) const {
      auto value_idx = getValueIndex(value);
      while (values[value_idx].occupied) {
        if (values[value_idx].value == value) {
          return keys[values[value_idx].key_index].key;
        }
        value_idx = (value_idx + 1) % capacity;
      }
      return std::nullopt;
    }

    ODIN_INLINE bool contains_key(const key_type& key) const { return getValueOf(key).has_value(); }

    ODIN_INLINE bool contains_value(const value_type& value) const {
      return getKeyOf(value).has_value();
    }

    ODIN_INLINE std::optional<value_type> getValueOf(const key_type& key) const {
      auto key_idx = getKeyIndex(key);
      while (keys[key_idx].occupied) {
        if (keys[key_idx].key == key) {
          return values[keys[key_idx].value_index].value;
        }
        key_idx = (key_idx + 1) % capacity;
      }
      return std::nullopt;
    }

    size_t size() const { return current_size; }

    size_t max_size() const { return capacity; }

    iterator begin() { return iterator(keys.get(), values.get(), 0, capacity); }

    iterator end() {
      return iterator(keys.get(), values.get(), capacity, capacity);
    }
};
