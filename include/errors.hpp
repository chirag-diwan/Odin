#pragma once

#include "./logging.hpp"
#include <cstdlib>

template <typename ...Pack>
ODIN_INLINE void Errorif(bool condition , Pack ... args ){
  if(__builtin_expect(condition , false)){
    Log(ERROR , args...);
    std::exit(-1);
  }
}
