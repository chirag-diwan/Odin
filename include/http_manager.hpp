#pragma once

#include <string>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "./data_structures/unidirectional_map.hpp"
#include "./data_structures/lock_free_ring_buffer.hpp"

#include "../external/simdjson/simdjson.h"
#define CPPHTTPLIB_NO_MULTI_THREAD_SUPPORT
#include "../external/httplib/httplib.h"


struct PromptReq{
  std::string content;
  std::string role;
};

class HttpManager{
  private:
    simdjson::dom::parser jsonParser_;
    
    const std::vector<const char *> filePaths_ = {
      "/index.html",
      "/style.css",
      "/dist/main.js",
    };

    unidirectional_map<std::string, std::string> fileContent_;

    httplib::Server server_;

    short port_;

    std::thread handler_;

    ringbuffer<std::string> infered_;
    std::condition_variable inferedCv_;
    std::mutex inferedMutex_;

    ringbuffer<PromptReq> prompts_;
    std::condition_variable readCv_;
    std::mutex promptMutex_;

    std::atomic<bool> isRunning_ = true;

    std::shared_ptr<std::sig_atomic_t> interupt_;

    void genericHandler(const httplib::Request& request , httplib::Response& response);
    void tokenStreamHandler(const httplib::Request&, httplib::Response& res);
    void tokenOneshotHandler(const httplib::Request& request , httplib::Response& response );
    void promptIncomeHandler(const httplib::Request& req , httplib::Response& );
  
    uint32_t promptTokens_;

  public:
    const inline static std::string DONE_TOK = "data: [DONE]\n\n";

    void Init(std::shared_ptr<std::sig_atomic_t> intrpt , short port = 8080);

    void StartListen();

    PromptReq ReadPrompt();

    void SetPromptTokenCount(uint32_t tok_count){
      promptTokens_ = tok_count;
    }

    bool WriteInfered(const std::string& tok);

    void Stop();
};
