#pragma once
#include <array>
#include <cstddef>
#include <string_view>

template <size_t max_size>
class ConstexprStr{
  private:
    std::array<char , max_size> data;
    size_t size;
  public:
    constexpr std::string_view view() const {
      return {data.data() , size};
    }

    constexpr char& operator[](size_t idx){
      return data[idx];
    }

    constexpr void set_size(size_t s){
      size = s;
    }
};
