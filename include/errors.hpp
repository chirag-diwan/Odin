#pragma once

#include "./logging.hpp"
#include <cstdlib>

template <typename ...Pack>
__attribute__((always_inline)) inline void Errorif(bool condition , Pack ... args ){
  if(__builtin_expect(condition , false)){
    Log(ERROR , args...);
    std::exit(-1);
  }
}
