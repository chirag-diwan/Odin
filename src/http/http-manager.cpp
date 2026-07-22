#include <atomic>
#include <filesystem>
#include <optional>
#include <string_view>
#include <format>
#include "../../external/nlohmann/json.hpp"
#include "../../external/simdjson/simdjson.h"
#define CPPHTTPLIB_NO_MULTI_THREAD_SUPPORT
#include "../../include/http-manager.hpp"
#include "../../include/logging.hpp"

std::string escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);

  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\""; break;
      case '\\':
        out += "\\\\"; break;
      case '\b':
        out += "\\b";  break;
      case '\f':
        out += "\\f";  break;
      case '\n':
        out += "\\n";  break;
      case '\r':
        out += "\\r";  break;
      case '\t':
        out += "\\t";  break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[7];
          snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
          out += buf;
        } else {
          out += c;
        }
    }
  }

  return out;
}

void HttpManager::generic_handler(const httplib::Request& request , httplib::Response& response){
  std::string path;
  if(request.path == "/"){
    path = "/index.html";
  }else{
    path = request.path;
  }

  std::string content = *file_content_.getValueOf(path);

  if(path.ends_with(".js")){
    response.set_content(content.c_str() , content.size(), "text/javascript");
  }else if(path.ends_with(".css")){
    response.set_content(content.c_str() , content.size(), "text/css");
  }else if(path.ends_with(".html")){
    response.set_content(content.c_str() , content.size(), "text/html");
  }
}

void HttpManager::token_stream_handler(const httplib::Request& _, httplib::Response& res){
  res.set_header("Content-Type", "text/event-stream");
  res.set_header("Connection", "keep-alive");
  res.set_chunked_content_provider("text/event-stream", [this](size_t /*offset*/, httplib::DataSink& sink) ->bool{ 
      while(is_running_){
      {
      std::unique_lock<std::mutex> lck(infered_mutex_);
      infered_cv_.wait(lck, [this] {
          return (interupt_ || !is_running_ || !infered_.empty());
          });
      }

      if(!is_running_){
        break;
      }

      if(interupt_){
        break;
      }

      if(infered_.empty()) continue;

      auto tok = *infered_.pop();

      if(tok == DONE_TOK){
        sink.write(DONE_TOK.data() , DONE_TOK.size());
        return true;
      }

      nlohmann::json response_json = {
        {"object", "chat.completion.chunk"},
        {"choices", nlohmann::json::array({
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

      return true;
  });
}

void HttpManager::token_oneshot_handler(const httplib::Request& _ , httplib::Response& response ){
  response.set_header("Content-Type", "application/json");

  std::string final_tok_string;final_tok_string.reserve(infered_.size() * 5);
  uint32_t tok_count = 0;
  while(is_running_){
    {
      std::unique_lock<std::mutex> lck(infered_mutex_);
      infered_cv_.wait(lck, [this] {
          return (interupt_ || !is_running_ || !infered_.empty());
          });
    }

    if(!is_running_){
      break;
    }

    if(interupt_){
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
    {"choices", nlohmann::json::array({
        {
        {"index", 0},
        {"message", {
        {"role", "assistant"},
        {"content", final_tok_string}
        }},
        {"finish_reason", "stop"}
        }
        })},
    {"usage", {
                {"prompt_tokens", prompt_tokens_},
                {"completion_tokens", tok_count},
                {"total_tokens", prompt_tokens_ + tok_count}
              }}
  };

  response.set_content(response_json.dump(), "application/json");
}

void HttpManager::prompt_income_handler(const httplib::Request& request , httplib::Response& response ){
  auto dom = json_parser_.parse(request.body.data(), request.body.size());

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
    Role role = Role::USER;

    if(buf == "system"){
      role = Role::SYSTEM;
    }

    auto ret = prompts_.push({
        .content = content ,
        .role = role,
        });

    if(!ret){
      Log(WARN , "Push to prompt failed");
    }else{
      read_cv_.notify_all();
    }
  }

  if(stream){
    token_stream_handler(request, response);
  }else{
    token_oneshot_handler(request, response);
  }
}

HttpManager::HttpManager(std::sig_atomic_t& intrpt , short port) : port_(port) ,is_running_(true) ,interupt_(intrpt){
  std::string root_abs = std::filesystem::absolute("./interface");
  file_content_.populate(file_paths_.size());
  if(!std::filesystem::is_directory(root_abs)){
    Log(ERROR ,"Frontend interface not present in path", root_abs);
    return;
  }

  std::ifstream in;
  for(const auto& file_path : file_paths_){
    auto abs_file_path = root_abs + file_path;
    in.open(abs_file_path);
    std::string content(std::istreambuf_iterator<char>{in} , std::istreambuf_iterator<char>{});
    in.close();
    file_content_.insert(file_path, content);
  }


  server_.Get("/", [this](const httplib::Request& request , httplib::Response& response) {
      generic_handler(request, response);
      });

  for(const auto& path : file_paths_){
    server_.Get(path, [this](const httplib::Request& request , httplib::Response& response) {
        generic_handler(request, response);
        });
  }

  server_.Post("/v1/chat/completions", [this](const httplib::Request& request , httplib::Response& response) {
      prompt_income_handler(request, response);
      });
}

void HttpManager::start_listen(){
  Log(INFO, std::format("Listening on http://localhost:{}" , port_));
  handler_ = std::thread([this](){
      server_.listen("localhost", port_);
      });
}

PromptReq HttpManager::read_prompt() {
  std::unique_lock<std::mutex> lock(prompt_mutex_);
  while(true){
    bool got_data = read_cv_.wait_for(lock, std::chrono::milliseconds(500), [this] {
        return !is_running_ || !prompts_.empty() ;
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

  if (prompts_.empty()) return {};
  return *prompts_.pop();
}


bool HttpManager::write_infered(const std::string& tok){
  auto ok = infered_.push(tok);
  if(ok){
    infered_cv_.notify_one();
  }

  return ok;
}


void HttpManager::stop(){
  is_running_ = false;

  infered_cv_.notify_all();
  read_cv_.notify_all();
  server_.stop();
  handler_.join();
}
