#pragma once

#include "ggml.h"
#include <vector>


struct Page{
  static constexpr uint8_t PAGE_TOKEN_SPAN = 32;
  ggml_tensor* data;
};

struct Cache{
  std::vector<Page> pages;
  uint8_t token_filled; // For the last Page
  Cache() : token_filled(0){}
};
