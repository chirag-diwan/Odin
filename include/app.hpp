#pragma once
#include "engine.hpp"
#include "ipc_manager.hpp"
#include "json_tokeniser.hpp"
#include "ggufparser.hpp"
#include "types.hpp"
#include "formatter.hpp"
#include "http_manager.hpp"
#include "../external/replxx/include/replxx.hxx"
#include "../external/ggml/include/ggml.h"
#include "../external/ggml/include/ggml-alloc.h"
#include <csignal>
#include <cstdlib>
#include <memory>
#include <string>
#include <sys/mman.h>

namespace odin{
  class App{
    private:
      static inline std::shared_ptr<std::sig_atomic_t> interupt = std::make_shared<std::sig_atomic_t>(false);

      static constexpr ggml_init_params staticCtxParams = {
        .mem_size = 10 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true 
      };

      static void sig_int_handler(int){
        *interupt = true;
      }

      void openFileMmap(const std::string& filepath);

      void initGGml();

      void initKVCaches();

      GGufParser parser;

      int mmapFD;
      void* mmapFilePtr;
      size_t mmapFileSize;

      ggml_backend* ggmlBackend;
      ggml_threadpool* ggmlThreadpool;
      ggml_context* ggmlContext;
      ggml_gallocr* prefillAllocr;
      ggml_gallocr* inferAllocr;

      ggml_tensor* kCache;
      ggml_tensor* vCache;

      ggml_backend_buffer* kvBuffer;

      Model model;

      Engine engine;
      BPETokeniser tokeniser;

      TemplateParamGenerator tpgenerator;
      Formatter formatter;

      replxx::Replxx rx;

      HttpManager httpManager;
      IPCManager ipcManager;

      enum class AppType{
        CHAT,
        HTTP_SERVER,
        IPC_SERVER
      };

      AppType appType;

    public:
      void InitBase(const Config& conf);

      void ConfigureChat();

      void ConfigureHttpServer();

      void ConfigureIPCServer();

      void Run();

      void Delete();
  };
}
