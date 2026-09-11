#include <atomic>
#include <filesystem>
#include <optional>
#include <string_view>
#include <format>
#include "nlohmann/json.hpp"
#include "simdjson/simdjson.h"
#include "logging.hpp"

#define CPPHTTPLIB_NO_MULTI_THREAD_SUPPORT
#include "http_manager.hpp"

void HttpManager::genericHandler(const httplib::Request& request , httplib::Response& response){
  Log(INFO, std::format("[{}] {} from {}", request.method, request.path, request.remote_addr));

  std::string path;
  if(request.path == "/"){
    path = "/index.html";
  }else{
    path = request.path;
  }

  std::string content = *fileContent_.getValueOf(path);

  if(path.ends_with(".js")){
    response.set_content(content.c_str() , content.size(), "text/javascript");
  }else if(path.ends_with(".css")){
    response.set_content(content.c_str() , content.size(), "text/css");
  }else if(path.ends_with(".html")){
    response.set_content(content.c_str() , content.size(), "text/html");
  }
}

void HttpManager::tokenStreamHandler(const httplib::Request& _, httplib::Response& res){
  Log(INFO, "[POST] /v1/chat/completions - streaming response");

  res.set_header("Content-Type", "text/event-stream");
  res.set_chunked_content_provider("text/event-stream", [this](size_t /*offset*/, httplib::DataSink& sink) ->bool{ 
    while(isRunning_){
      {
        std::unique_lock<std::mutex> lck(inferedMutex_);
        inferedCv_.wait(lck, [this] {
          return (*interupt_ || !isRunning_ || !infered_.empty());
        });
      }

      if(!isRunning_){
        break;
      }

      if(*interupt_){
        break;
      }

      if(infered_.empty()) continue;

      auto tok = *infered_.pop();

      if(tok == DONE_TOK){
        sink.write(DONE_TOK.data() , DONE_TOK.size());
        sink.done();  
        return false;
      }

      nlohmann::json response_json = {
        {"object", "chat.completion.chunk"},
        {"choices",
          nlohmann::json::array({
            {
              {"index", 0},
              {
                "delta", {
                  {"role", "assistant"},
                  {"content", tok}
                }
              },
              {"finish_reason", nullptr}
            }
          })}
      };


      auto msg = std::format("data: {}\n\n", response_json.dump());
      sink.write(msg.data(), msg.size());
    }

    return false;
  });
}

void HttpManager::tokenOneshotHandler(const httplib::Request& _ , httplib::Response& response ){
  Log(INFO, "[POST] /v1/chat/completions - non-streaming response");

  response.set_header("Content-Type", "application/json");

  std::string final_tok_string;final_tok_string.reserve(infered_.size() * 5);
  uint32_t tok_count = 0;
  while(isRunning_){
    {
      std::unique_lock<std::mutex> lck(inferedMutex_);
      inferedCv_.wait(lck, [this] {
        return (*interupt_ || !isRunning_ || !infered_.empty());
      });
    }

    if(!isRunning_){
      break;
    }

    if(*interupt_){
      break;
    }

    if(infered_.empty()) continue;

    auto tok = *infered_.pop();
    if(tok == DONE_TOK){
      break;
    }

    final_tok_string.append(tok.data(), tok.size());
    tok_count ++;
  }

  nlohmann::json response_json = {
    {"object", "chat.completion"},
    {"choices",
      nlohmann::json::array({
        {
          {"index", 0},
          {"message", {
                        {"role", "assistant"},
                        {"content", final_tok_string}
                      }},
          {"finish_reason", "stop"}
        }
      })},
      {"usage",
        {
          {"prompt_tokens", promptTokens_},
          {"completion_tokens", tok_count},
          {"total_tokens", promptTokens_ + tok_count}
        }}
  };


  response.set_content(response_json.dump(), "application/json");
}

void HttpManager::promptIncomeHandler(const httplib::Request& request , httplib::Response& response ){
  Log(INFO, std::format("[POST] {} from {}", request.path, request.remote_addr));

  auto dom = jsonParser_.parse(request.body.data(), request.body.size());

  std::string_view buf;
  simdjson::dom::element value;

  bool stream = true;

  auto status = dom["stream"].get(value);
  if(status == simdjson::SUCCESS){
    stream = value.get_bool();
  }


  status = dom["messages"].get(value);
  if(status != simdjson::SUCCESS){

    response.status = 400;
    auto res = nlohmann::json({{"error", {{"message", "Empty messages are not allowed"}, {"type", "invalid_request_error"}}}}).dump();
    response.set_content(res.data() , res.size() , "application/json");
    return;
  }

  for(const auto& msg_obj : value.get_array()){
    status = msg_obj["content"].get(value);
    if(status != simdjson::SUCCESS){

      response.status = 400;
      auto res = nlohmann::json({{"error", {{"message", "JSON parsing error , content field not set"}, {"type", "invalid_request_error"}}}}).dump();
      response.set_content(res.data() , res.size() , "application/json");
      return;
    }

    buf = value.get_string();
    std::string content{buf.data(), buf.size()};

    status = msg_obj["role"].get(value);
    if(status != simdjson::SUCCESS){

      response.status = 400;
      auto res = nlohmann::json({{"error", {{"message", "JSON parsing error , content field not set"}, {"type", "invalid_request_error"}}}}).dump();
      response.set_content(res.data() , res.size() , "application/json");
      return;
    }

    buf = value.get_string();
    auto ret = prompts_.push({
      .content = content ,
      .role = std::string{buf},
    });

    if(!ret){
      Log(INFO, "Push to prompt failed");
    }else{
      readCv_.notify_all();
    }
  }

  if(stream){
    tokenStreamHandler(request, response);
  }else{
    tokenOneshotHandler(request, response);
  }
}

void HttpManager::Init(std::shared_ptr<std::sig_atomic_t> intrpt , short port){
  port_ = port;
  isRunning_ = true;
  interupt_ = intrpt;

  Log(INFO, std::format("Initializing HTTP server on port {}", port_));

  std::string root_abs = std::filesystem::absolute("./interface");
  fileContent_.populate(filePaths_.size());
  if(!std::filesystem::is_directory(root_abs)){
    Log(INFO, std::format("Frontend interface not present in path: {}", root_abs));
    return;
  }

  Log(INFO, std::format("Loading frontend interface from {}", root_abs));

  std::ifstream in;
  for(const auto& file_path : filePaths_){
    auto abs_file_path = root_abs + file_path;
    in.open(abs_file_path);
    std::string content(std::istreambuf_iterator<char>{in} , std::istreambuf_iterator<char>{});
    in.close();
    auto _ = fileContent_.insert(file_path, content);
    Log(INFO, std::format("Loaded {} ({} bytes)", file_path, content.size()));
  }


  server_.Get("/", [this](const httplib::Request& request , httplib::Response& response) {
    genericHandler(request, response);
  });

  for(const auto& path : filePaths_){
    server_.Get(path, [this](const httplib::Request& request , httplib::Response& response) {
      genericHandler(request, response);
    });
  }

  server_.Post("/v1/chat/completions", [this](const httplib::Request& request , httplib::Response& response) {
    Log(INFO, "[POST] /v1/chat/completions");
    promptIncomeHandler(request, response);
  });

  Log(INFO, "HTTP server routes initialized");
}

void HttpManager::StartListen(){
  Log(INFO, std::format("Listening on http://localhost:{}" , port_));
  handler_ = std::thread([this](){
    server_.listen("localhost", port_);
    Log(INFO, "HTTP server stopped listening");
  });
}

PromptReq HttpManager::ReadPrompt() {
  std::unique_lock<std::mutex> lock(promptMutex_);
  while(true){
    bool got_data = readCv_.wait_for(lock, std::chrono::milliseconds(500), [this] {
      return !isRunning_ || !prompts_.empty() ;
    });

    if (!got_data) {
      if (*interupt_) {
        return {}; 
      }

      continue;
    }else{
      break;
    }
  }

  if (prompts_.empty()) return {};

  return *prompts_.pop();
}


bool HttpManager::WriteInfered(const std::string& tok){
  auto ok = infered_.push(tok);

  if(ok){
    inferedCv_.notify_one();
  }else{
    Log(INFO, "Failed to push inference token");
  }

  return ok;
}


void HttpManager::Stop(){
  Log(INFO, "Stopping HTTP server");

  isRunning_ = false;

  inferedCv_.notify_all();
  readCv_.notify_all();
  server_.stop();
  handler_.join();

  Log(INFO, "HTTP server stopped");
}
