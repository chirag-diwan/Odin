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
    std::cout << "[WARN]";
  }else if(l == ERROR) {
    std::cout << "[ERROR]";
  }else if(l == INFO) {
    std::cout << "[INFO]";
  }
  ((std::cout << args << ' '), ...);
  std::cout << '\n';
}


template<typename ...Pack>
__attribute__((always_inline)) inline void Log(Pack ...args){
  ((std::cout << args << ' '), ...);
  std::cout << '\n';
}
