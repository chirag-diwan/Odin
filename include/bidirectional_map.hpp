#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <optional>

struct bid_key_t {
  std::string_view key;
  std::size_t value_index;
  bool occupied = false;
};

struct bid_value_t {
  int32_t value;
  std::size_t key_index;
  bool occupied = false;
};

class bidirectional_map {
  private:
    std::unique_ptr<bid_key_t[]> keys;
    std::unique_ptr<bid_value_t[]> values;
    size_t capacity;
    size_t current_size;

    size_t getKeyIndex(std::string_view key) const {
      return std::hash<std::string_view>()(key) % capacity;
    }

    size_t getValueIndex(int32_t value) const {
      return std::hash<int32_t>()(value) % capacity;
    }

  public:
    explicit bidirectional_map(size_t max_elements) {
      if (max_elements == 0) {
        return;
      }
      capacity = max_elements * 2;
      current_size = 0;

      keys = std::make_unique<bid_key_t[]>(capacity);
      values = std::make_unique<bid_value_t[]>(capacity);
    }

    bool insert(std::string_view key, int32_t value) {
      if (current_size >= capacity / 2) {
        return false; 
      }

      if (getValueOf(key).has_value() || getKeyOf(value).has_value()) {
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

      keys[key_idx] = bid_key_t{.key = key, .value_index = value_idx, .occupied = true};
      values[value_idx] = bid_value_t{.value = value, .key_index = key_idx, .occupied = true};
      current_size++;

      return true;
    }

    std::optional<std::string_view> getKeyOf(int32_t value) const {
      auto value_idx = getValueIndex(value);
      while (values[value_idx].occupied) {
        if (values[value_idx].value == value) {
          return keys[values[value_idx].key_index].key;
        }
        value_idx = (value_idx + 1) % capacity;
      }
      return std::nullopt;
    }

    std::optional<int32_t> getValueOf(std::string_view key) const {
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
    };

    size_t max_size() const {
      return capacity;
    };
};
