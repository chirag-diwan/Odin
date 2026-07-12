#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdlib>
#include <memory>
#include <sys/mman.h>

#include <cstddef>
class MmapGuard {
  void* addr;
  size_t len;
  public:
  MmapGuard(void* addr, size_t len) : addr(addr), len(len) {}
  ~MmapGuard() { if (addr) munmap(addr, len); }
};

struct ThreadPoolDeleter{
  void operator()(ggml_threadpool* ctx) const { if (ctx) ggml_threadpool_free(ctx); }
};


using UniqueThreadpool = std::unique_ptr<ggml_threadpool, ThreadPoolDeleter>;

struct GgmlDeleter {
  void operator()(ggml_context* ctx) const { if (ctx) ggml_free(ctx); }
};

using UniqueGgmlContext = std::unique_ptr<ggml_context, GgmlDeleter>;

