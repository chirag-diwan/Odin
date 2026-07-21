#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>
#define CPPHTTPLIB_NO_MULTI_THREAD_SUPPORT
#include "../external/simdjson/simdjson.h"
#include "../external/httplib/httplib.h"
#include "./data_structures/unidirectional_map.hpp"
#include "./data_structures/lock_free_ring_buffer.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>

enum class Role : uint8_t {
  SYSTEM,
  USER
};

struct PromptReq{
  std::string content;
  Role role;
};

class HttpManager{
  private:
    simdjson::dom::parser json_parser_;
    
    const std::vector<const char *> file_paths_ = {
      "/index.html",
      "/style.css",
      "/dist/main.js",
    };

    unidirectional_map<std::string, std::string> file_content_;

    httplib::Server server_;

    std::thread handler_;

    ringbuffer<std::string> infered_;
    std::condition_variable infered_cv_;
    std::mutex infered_mutex_;

    ringbuffer<PromptReq> prompts_;
    std::condition_variable read_cv_;
    std::mutex prompt_mutex_;

    std::atomic<bool> is_running_ = true;

    std::sig_atomic_t& interupt_;

    void generic_handler(const httplib::Request& request , httplib::Response& response);
    void token_stream_handler(const httplib::Request&, httplib::Response& res);
    void token_oneshot_handler(const httplib::Request& request , httplib::Response& response );
    void prompt_income_handler(const httplib::Request& req , httplib::Response& );

  public:
    const inline static std::string DONE_TOK = "data: [DONE]\n\n";

    HttpManager(std::sig_atomic_t& intrpt);

    void start_listen();

    PromptReq read_prompt();

    bool write_infered(const std::string& tok);

    void stop();
};
