#pragma once
#include <iostream>
enum LogLevel {
  WARN,
  ERROR,
  INFO
};

template<typename ...Pack>
__attribute__((always_inline)) inline void Log(LogLevel l , Pack ...args){
  if(l == WARN){
    std::cerr << "[WARN]";
  }else if(l == ERROR) {
    std::cerr << "[ERROR]";
  }else if(l == INFO) {
    std::cerr << "[INFO]";
  }
  ((std::cerr << args << ' '), ...);
  std::cerr << '\n';
}


template<typename ...Pack>
__attribute__((always_inline)) inline void Log(Pack ...args){
  ((std::cerr << args << ' '), ...);
  std::cerr << '\n';
}
