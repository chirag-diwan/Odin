#include "../include/app.hpp"
#include "../include/model_utils.hpp"
#include "../external/ggml/include/ggml-cpu.h"
#include "../external/ggml/include/ggml-backend.h"
#include <cstdint>
#include <iostream>
#include <thread>

namespace odin{
  void App::openFileMmap(const std::string& filepath){
    mmapFD = open(filepath.c_str(), O_RDONLY);
    Errorif(mmapFD == -1, "Not a valid file descriptor for %?", filepath);

    struct stat file_statistics;
    Errorif(fstat(mmapFD, &file_statistics) == -1, "Unable to get file stats for ", filepath);
    mmapFileSize = file_statistics.st_size;

    mmapFilePtr = mmap(NULL, file_statistics.st_size, PROT_READ, MAP_PRIVATE, mmapFD, 0);
    Errorif(mmapFilePtr == MAP_FAILED, "Mapping failed for ", filepath);
  }

  void App::initGGml(){
    ggmlBackend = ggml_backend_cpu_init();
    auto threadpool_params = ggml_threadpool_params_default(std::thread::hardware_concurrency());
    ggmlThreadpool = ggml_threadpool_new(&threadpool_params);
    ggml_backend_cpu_set_threadpool(ggmlBackend, ggmlThreadpool);
    ggmlContext = ggml_init(staticCtxParams);
    prefillAllocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ggmlBackend));
    inferAllocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ggmlBackend));
  }

  void App::initKVCaches(){
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

  void App::InitBase(const Config& conf){
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
  }

  void App::ConfigureChat(){
    appType = AppType::CHAT;

    rx.install_window_change_handler();
    rx.set_max_history_size(1000);

    rx.bind_key(replxx::Replxx::KEY::ENTER, [this](char32_t) {
      rx.invoke(replxx::Replxx::ACTION::INSERT_CHARACTER, '\n');
      return replxx::Replxx::ACTION_RESULT::CONTINUE;
    });

    rx.bind_key(replxx::Replxx::KEY::control('S'), [](char32_t) {
      return replxx::Replxx::ACTION_RESULT::RETURN;
    });
  }

  void App::ConfigureHttpServer(){
    appType = AppType::HTTP_SERVER;
    httpManager.Init(interupt);
  }

  void App::ConfigureIPCServer(){
    appType = AppType::IPC_SERVER;
    ipcManager.Init(interupt);
  }

  void App::Run(){
    std::vector<uint32_t> tokens{};
    std::string system_prompt = "You are a helpful, accurate, and concise AI assistant. You have to respond in the tool format only when you need to use a tool , respond in plain english without format otherwise";

    std::string raw_prompt;

    bool is_first = true;

    if(appType == AppType::CHAT){
      rx.clear_screen();
    }

    if(appType == AppType::HTTP_SERVER){
      httpManager.StartListen();
    }

    if(appType == AppType::IPC_SERVER){
      ipcManager.StartListen();
    }

    while (!(*interupt)) {
      tpgenerator.Reset();
      tpgenerator.SetDefault(model.globals.generalModelArchitecture);
      if(is_first){
        is_first = false;
        tpgenerator.SetTools();
      }

      if(appType == AppType::CHAT){
        const char* c_input = rx.input("\n $ ");
        if (!c_input) {
          break;
        }

        raw_prompt = c_input;

        if (raw_prompt.starts_with("!exit")) break;
        if(raw_prompt.starts_with("!system")) {
          system_prompt = raw_prompt.substr(7);
        }
      }else if(appType == AppType::HTTP_SERVER){
        auto prompt_req = httpManager.ReadPrompt();
        raw_prompt = prompt_req.content;

        if(prompt_req.role == "system"){
          system_prompt = raw_prompt;
          continue;
        }
      }else if(appType == AppType::IPC_SERVER){
        raw_prompt = ipcManager.ReadPrompt();
      }

      if (raw_prompt.empty()) {
        continue;
      }

      rx.history_add(raw_prompt);

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
      if(appType == AppType::HTTP_SERVER){
        httpManager.SetPromptTokenCount(span_size);
      }

      uint32_t next_token = engine.Prefill(tokens_view);
      tokens.push_back(next_token);

      auto tok = tokeniser.Decode(next_token);

      if(tok.has_value()){
        switch (appType) {
          case AppType::CHAT:
            std::cerr << *tok;
            break;
          case AppType::HTTP_SERVER:
            httpManager.WriteInfered(*tok);
            break;
          case AppType::IPC_SERVER:
            ipcManager.WriteInfered(*tok);
            break;
        }
      }

      while (!*interupt && (next_token != model.globals.ggmlEosTokenId)) {

        next_token = engine.Infer(tokens.back());
        tokens.push_back(next_token);

        if (next_token != model.globals.ggmlEosTokenId) {
          auto tok = tokeniser.Decode(next_token);
          if(tok.has_value()){
            switch (appType) {
              case AppType::CHAT:
                std::cerr << *tok;
                break;
              case AppType::HTTP_SERVER:
                httpManager.WriteInfered(*tok);
                break;
              case AppType::IPC_SERVER:
                ipcManager.WriteInfered(*tok);
                break;
            }
          }
          continue;
        }

        break;
      }

      if(appType == AppType::HTTP_SERVER){
        httpManager.WriteInfered(httpManager.DONE_TOK);
      }

      *interupt = false;
    }

    if(appType == AppType::HTTP_SERVER){
      httpManager.Stop();
    }

    if(appType == AppType::IPC_SERVER){
      ipcManager.Stop();
    }
  }

  void App::Delete(){
    ipcManager.Delete();
    tokeniser.Delete();

    ggml_backend_buffer_free(kvBuffer);
    ggml_gallocr_free(inferAllocr);
    ggml_gallocr_free(prefillAllocr);
    ggml_threadpool_free(ggmlThreadpool);
    ggml_backend_free(ggmlBackend);

    munmap(mmapFilePtr, mmapFileSize);
  }
}
