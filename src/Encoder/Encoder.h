#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace Odin {
struct TokenPair {
  uint32_t First;
  uint32_t Second;

  bool operator==(const TokenPair& other) const {
    return First == other.First && Second == other.Second;
  }
};

void BPEEncoder(const std::string&                         prompt,
                std::unordered_map<uint32_t, std::string>& vocab,
                std::vector<uint32_t>&                     tokens);
}; // namespace Odin

template <> struct std::hash<Odin::TokenPair> {
  std::size_t operator()(const Odin::TokenPair& c) const {
    std::size_t h1 = std::hash<int>{}(c.First);
    std::size_t h2 = std::hash<short>{}(c.Second);

    return h1 ^ (h2 << 1);
  }
};
