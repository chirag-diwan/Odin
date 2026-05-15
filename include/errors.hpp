#pragma once

#include "./logging.hpp"
#include <cstdlib>

template <typename ...Pack>
void Errorif(bool condition , Pack ... args ){
  if(condition){
    Log(ERROR , args...);
    std::exit(-1);
  }
}
