#pragma once
#include "engine.hpp"
#include "ggml-backend.h"
#include "model_utils.hpp"
#include "json_tokeniser.hpp"
#include "ggufparser.hpp"
#include "../external/replxx/include/replxx.hxx"
#include "types.hpp"
#include "formatter.hpp"
#include "../external/ggml/include/ggml.h"
#include "../external/ggml/include/ggml-alloc.h"
#include "../external/ggml/include/ggml-cpu.h"
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <sys/mman.h>
#include <thread>

namespace odin{
  class App{
    private:
      static inline std::sig_atomic_t interupt = false;

      static constexpr ggml_init_params staticCtxParams = {
        .mem_size = 10 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = true 
      };

      static void sig_int_handler(int){
        interupt = true;
      }

      void openFileMmap(const std::string& filepath){
        mmapFD = open(filepath.c_str(), O_RDONLY);
        Errorif(mmapFD == -1, "Not a valid file descriptor for %?", filepath);

        struct stat file_statistics;
        Errorif(fstat(mmapFD, &file_statistics) == -1, "Unable to get file stats for ", filepath);
        mmapFileSize = file_statistics.st_size;

        mmapFilePtr = mmap(NULL, file_statistics.st_size, PROT_READ, MAP_PRIVATE, mmapFD, 0);
        Errorif(mmapFilePtr == MAP_FAILED, "Mapping failed for ", filepath);
      }

      void initGGml(){
        ggmlBackend = ggml_backend_cpu_init();
        auto threadpool_params = ggml_threadpool_params_default(std::thread::hardware_concurrency());
        ggmlThreadpool = ggml_threadpool_new(&threadpool_params);
        ggml_backend_cpu_set_threadpool(ggmlBackend, ggmlThreadpool);
        ggmlContext = ggml_init(staticCtxParams);
        prefillAllocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ggmlBackend));
        inferAllocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ggmlBackend));
      }

      void initKVCaches(){
        auto d_head = model.globals.embeddingLength / model.globals.attentionHeadCount;

        auto c = model.globals.contextLength; 
        auto n_head_kv = model.globals.attentionHeadCountKv;

        kCache = ggml_new_tensor_4d(ggmlContext, GGML_TYPE_F16, 
                                    d_head, c, n_head_kv , model.globals.blockCount);

        vCache = ggml_new_tensor_4d(ggmlContext, GGML_TYPE_F16, 
                                    c, d_head, n_head_kv , model.globals.blockCount);

        kvBuffer = ggml_backend_alloc_ctx_tensors(ggmlContext, ggmlBackend);
        Errorif(kvBuffer == nullptr, "Failed to allocate physical memory for KV cache");
      }

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

    public:
      App(){}

      void Init(const Config& conf){
        std::signal(SIGINT , sig_int_handler);

        openFileMmap(conf.modelPath);
        parser.ParseFile(mmapFD, mmapFilePtr, mmapFileSize);

        initGGml();

        model = CreateModel(ggmlContext, parser);

        initKVCaches();

        engine.Init(model, inferAllocr , prefillAllocr , ggmlBackend , kCache , vCache , kvBuffer);
        engine.ReservePrefillMemory();
        engine.ReserveDecodeMemory();

        tokeniser.Open(conf.tokeniserJsonPath);
        tpgenerator.SetDefault(model.globals.generalModelArchitecture);
        formatter.Init(std::string{model.globals.chat_template});

        rx.install_window_change_handler();
        rx.set_max_history_size(1000);

        rx.bind_key(
                    replxx::Replxx::KEY::ENTER,
                    [this](char32_t) {
                      rx.invoke(replxx::Replxx::ACTION::INSERT_CHARACTER, '\n');
                      return replxx::Replxx::ACTION_RESULT::CONTINUE;
                    }
                   );

        rx.bind_key(
                    replxx::Replxx::KEY::control('S'),
                    [](char32_t) {
                      return replxx::Replxx::ACTION_RESULT::RETURN;
                    }
                   );

      }

      void Run(){
        std::vector<uint32_t> tokens;
        std::string system_prompt = "You are a helpful, accurate, and concise AI assistant. You have to respond in the tool format only when you need to use a tool , respond in plain english without format otherwise";

        std::string raw_prompt;

        bool is_first = true;
        while (!interupt) {
          tpgenerator.Reset();
          if(is_first){
            is_first = false;
            tpgenerator.SetDefault(model.globals.generalModelArchitecture);
          }

          const char* c_input = rx.input("\n $ ");

          if (c_input == nullptr) {
            break;
          }else{
            raw_prompt = c_input;
          }

          if (raw_prompt.empty()) {
            continue;
          }

          rx.history_add(raw_prompt);

          if (raw_prompt.starts_with("!exit")) break;

          if(raw_prompt.starts_with("!system")) {
            system_prompt = raw_prompt.substr(7);
          }

          if(raw_prompt.starts_with("!clear-context")){
            engine.ClearContext();
          }

          tpgenerator.AddMessage("system", system_prompt);
          tpgenerator.AddMessage("user", raw_prompt);

          auto prompt = formatter.GetFormattedString(tpgenerator.GetRef());

          size_t last_index = tokens.size();
          tokeniser.Tokenise(prompt, tokens);

          size_t span_size = tokens.size() - last_index;
          std::span<uint32_t> tokens_view(tokens.data() + last_index, span_size);

          uint32_t next_token = engine.Prefill(tokens_view);
          tokens.push_back(next_token);

          auto tok = tokeniser.Decode(next_token);

          if(tok.has_value()){
            std::cerr << *tok;
          }

          while (!interupt && (next_token != model.globals.ggmlEosTokenId)) {

            next_token = engine.Infer(tokens.back());
            tokens.push_back(next_token);

            if (next_token != model.globals.ggmlEosTokenId) {

              auto tok = tokeniser.Decode(next_token);
              if(tok.has_value()){
                std::cerr << *tok;
              }
            }
          }

          interupt = false;
        }
      }

      void Delete(){
        tokeniser.Delete();

        ggml_backend_buffer_free(kvBuffer);
        ggml_gallocr_free(inferAllocr);
        ggml_gallocr_free(prefillAllocr);
        ggml_threadpool_free(ggmlThreadpool);
        ggml_backend_free(ggmlBackend);

        munmap(mmapFilePtr, mmapFileSize);
      }
  };
}
