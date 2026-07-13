#pragma once

#define CPPHTTPLIB_NO_MULTI_THREAD_SUPPORT
#include "../external/httplib/httplib.h"
#include "./data_structures/unidirectional_map.hpp"
#include "./data_structures/lock_free_ring_buffer.hpp"
#include <condition_variable>
#include <mutex>
#include <thread>

class HttpManager{
  private:
    std::string escape(std::string_view s) ;

    void generic_handler(const httplib::Request& request , httplib::Response& response);

    void token_stream_handler(const httplib::Request&, httplib::Response& res);

    void prompt_income_handler(const httplib::Request& req , httplib::Response& );

    const std::vector<const char *> file_paths = {
      "/index.html",
      "/style.css",
      "/dist/main.js",
    };

    unidirectional_map<std::string, std::string> file_content;

    httplib::Server server;

    std::thread handler_;

    ringbuffer<std::string> infered_;
    std::condition_variable infered_cv_;
    std::mutex infered_mutex_;

    ringbuffer<std::string> prompts_;
    std::condition_variable read_cv_;
    std::mutex prompt_mutex_;

    std::atomic<bool> is_running_ = true;

    std::sig_atomic_t& interupt_;

  public:
    const inline static std::string EOS = "data: [EOS]\n\n";

    HttpManager(std::sig_atomic_t& intrpt , const std::string& root = "../interface");

    void start_listen();

    std::string read_prompt() ;

    bool write_infered(const std::string& tok);

    void stop();
};
