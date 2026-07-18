#pragma once
#include <iostream>
enum LogLevel {
  WARN,
  ERROR,
  INFO
};

namespace ansi {
    constexpr const char* reset  = "\033[0m";
    constexpr const char* red    = "\033[31m";
    constexpr const char* yellow = "\033[33m";
    constexpr const char* green  = "\033[32m";
    constexpr const char* bold   = "\033[1m";
}

template<typename... Pack>
__attribute__((always_inline)) inline void Log(LogLevel l, Pack&&... args) {
  switch (l) {
    case INFO:
      std::cerr << ansi::bold << ansi::green << "[INFO]  " ;
      break;
    case WARN:
      std::cerr << ansi::bold << ansi::yellow << "[WARN]  ";
      break;
    case ERROR:
      std::cerr << ansi::bold << ansi::red << "[ERROR] ";
      break;
  }

  ((std::cerr << std::forward<Pack>(args) << ' '), ...);
  std::cerr << ansi::reset << '\n';
}


template<typename ...Pack>
__attribute__((always_inline)) inline void Log(Pack ...args){
  ((std::cerr << args << ' '), ...);
  std::cerr << '\n';
}
