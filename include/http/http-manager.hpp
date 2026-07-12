#pragma once

#include "../../external/httplib/httplib.h"
#include "../data_structures/unidirectional_map.hpp"
#include "../data_structures/lock_free_ring_buffer.hpp"
#include "../logging.hpp"
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

class HttpManager{
  private:
    httplib::Server server;

    const std::vector<const char * > file_paths = {
      "/index.html",
      "/style.css",
      "/dist/elements.js.map",
      "/dist/main.js.map",
      "/dist/server-side-event.js.map",
      "/dist/main.js",
      "/dist/elements.js",
      "/dist/server-side-event.js"
    };

    unidirectional_map<std::string, std::string> file_content;

    void generic_handler(const httplib::Request& request , httplib::Response& response){
      std::string path;
      if(request.path == "/"){
        path = "/index.html";
      }else{
        path = request.path;
      }
      Log(path);
      std::string content = *file_content.getValueOf(path);

      if(path.ends_with(".js")){
        response.set_content(content.c_str() , content.size(), "text/javascript");
      }else if(path.ends_with(".css")){
        response.set_content(content.c_str() , content.size(), "text/css");
      }else if(path.ends_with(".html")){
        response.set_content(content.c_str() , content.size(), "text/html");
      }
    }

    void token_stream_handler(const httplib::Request&, httplib::Response& res){
      res.set_header("Content-Type", "text/event-stream");
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Connection", "keep-alive");

      res.set_chunked_content_provider("text/event-stream",
            [this](size_t /*offset*/, httplib::DataSink& sink) {
                while (is_running_) {
                  std::unique_lock<std::mutex>lck(infered_mutex);
                  infered_cv_.wait(lck,[this]{
                    return !is_running_ || !infered.empty();
                  });
                  std::string msg = *infered.pop();
                  std::string sse_formatted = "data: " + msg + "\n\n";
                  sink.write(sse_formatted.data(), sse_formatted.size());
                }
                return false;
              }
          );
    }

    void prompt_income_handler(const httplib::Request& req , httplib::Response& ){
      auto ret = prompts.push(req.body);
      if(!ret){
        Log(WARN , "Push to prompt failed");
      }else{
        read_cv_.notify_all();
      }
    }

    std::thread handler_;
    ringbuffer<std::string> infered;
    std::condition_variable infered_cv_;
    std::mutex infered_mutex;



    ringbuffer<std::string> prompts;
    std::condition_variable read_cv_;
    std::mutex prompt_mutex;

    std::sig_atomic_t& interupt_;
    std::atomic<bool> is_running_ = true;

  public:

    HttpManager(std::sig_atomic_t& intrpt , const std::string& root = "../interface") : interupt_(intrpt) , is_running_(true){
      std::string root_abs = std::filesystem::absolute(root);
      file_content.populate(file_paths.size());

      std::ifstream in;
      for(const auto& file_path : file_paths){
        auto abs_file_path = root_abs + file_path;
        in.open(abs_file_path);
        std::string content(std::istreambuf_iterator<char>{in} , std::istreambuf_iterator<char>{});
        in.close();
        file_content.insert(file_path, content);
      }


      server.Get("/", [this](const httplib::Request& request , httplib::Response& response) {
          generic_handler(request, response);
          });

      for(const auto& path : file_paths){
        server.Get(path, [this](const httplib::Request& request , httplib::Response& response) {
            generic_handler(request, response);
            });

      }
      server.Get("/stream", [this](const httplib::Request& request , httplib::Response& response) {
          token_stream_handler(request, response);
          });

      server.Post("/prompt", [this](const httplib::Request& request , httplib::Response& response) {
          prompt_income_handler(request, response);
          });
    }

    void start_listen(){
      handler_ = std::thread([this](){
          server.listen("127.0.0.1", 8080);
          });
    }

    std::string read_prompt() {
      std::unique_lock<std::mutex> lock(prompt_mutex);
      while(true){
        bool got_data = read_cv_.wait_for(lock, std::chrono::milliseconds(500), [this] {
            return !is_running_ || !prompts.empty();
            });

        if (!got_data) {
          if (interupt_) {
            return {}; 
          }
          continue;
        }else{
          break;
        }
      }
      if (prompts.empty()) return {};
      return *prompts.pop();
    }


    bool write_infered(const std::string& tok){
      auto ok = infered.push(tok);
      if(ok){
        infered_cv_.notify_one();
      }
      return ok;
    }


    void stop(){
      is_running_ = false;

      infered_cv_.notify_all();
      read_cv_.notify_all();
      server.stop();
      handler_.join();
    }
};
