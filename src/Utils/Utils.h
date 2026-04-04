#pragma once
#include <cstdint>
#include <cstdlib>
#include <iostream>

template <class... lambdas> struct mix : lambdas... {
  using lambdas::operator()...;
};
template <class... lambdas> mix(lambdas...) -> mix<lambdas...>;

template <typename... Args> inline void WARNIF(bool condition, Args... args) {
  if (condition) {
    std::cout << __LINE__ << " WARN :: ";
    ((std::cout << args), ...);
    std::cout << '\n';
  }
}

template <typename... Args> inline void WARN(Args... args) {
  std::cout << __LINE__ << " WARN :: ";
  ((std::cout << args), ...);
  std::cout << '\n';
}

template <typename... Args> inline void ERROR(Args... args) {
  std::cout << __LINE__ << " ERROR :: ";
  ((std::cout << args), ...);
  std::cout << '\n';
}

template <typename... Args> inline void ERROR_AND_EXIT(Args... args) {
  std::cout << __LINE__ << " ERROR :: ";
  ((std::cout << args), ...);
  std::cout << '\n';
  std::exit(EXIT_FAILURE);
}

// ASSERT when condition is true
template <typename... Args>
inline constexpr void ERRORIF(bool Condition, Args... args) {
  if (Condition) {
    std::cout << __LINE__ << " ERROR :: ";
    ((std::cout << args), ...);
    std::cout << '\n';
    std::exit(EXIT_FAILURE);
  }
}
