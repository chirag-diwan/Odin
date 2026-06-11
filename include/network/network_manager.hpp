#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <string_view>
#include <thread>
#include "../data_structures/lock_free_ring_buffer.hpp"
#include "../logging.hpp"


class NetworkManager{
  private:
    static void handle_client(int server_fd , ringbuffer<std::string>& prompt , ringbuffer<std::string>& decode, std::atomic<bool>& stop_flag , std::condition_variable& condition_prompt_empty , std::mutex& prompt_mutex , std::condition_variable& condition_decode_empty , std::mutex& decode_mutex){
      std::string buffer; buffer.resize(1024);
      bool close_server = false;

      while(!close_server){
        auto data_socket = accept(server_fd, nullptr , nullptr);
        if(data_socket == -1 ){
          if(stop_flag)break;
          Log(ERROR,"Accept failed" , strerror(errno));
          continue;
        }
        int result;

        std::string prompt_str;

        while(true){

          memset(buffer.data(), 0, buffer.size());
          result = read(data_socket, buffer.data(), buffer.size());
          if(result == -1){
            Log(ERROR, "Read failed" , strerror(errno));
          }

          if(result == 0){
            break;
          }

          std::string_view buffer_view(buffer.data() , result);

          if(buffer_view == "NET_END") {
            break;
          }

          if(buffer_view == "NET_CLOSE") {
            close_server = true;
            break;
          }

          prompt_str.append(buffer.data() , result);
    
          std::unique_lock<std::mutex> lock(decode_mutex);
          condition_decode_empty.wait(lock , [&decode](){
              return !decode.empty();
              });
          while(!decode.empty()){
            auto decode_token = *decode.pop();
            send(data_socket,decode_token.data(),decode_token.size(), 0);
          }
        }
        close(data_socket);
        if(!prompt_str.empty()){
          bool ret = false;
          {
            std::lock_guard<std::mutex> lock(prompt_mutex);
            ret = prompt.push(prompt_str) ;
          }
          if(!ret){
            Log(ERROR, "Push failed to prompt" , strerror(errno));
          }else{
            condition_prompt_empty.notify_one();
          }
        }
      }
    }

    std::mutex prompt_mutex_;
    std::condition_variable condition_prompt_empty_;

    std::mutex decode_mutex_;
    std::condition_variable condition_decode_empty_;

    const std::string path_;


    ringbuffer<std::string> prompt;
    ringbuffer<std::string> infered;

    std::thread client_handlers;

    std::atomic<bool> stop_ = false;
  public:

    int server_socket = -1;
    NetworkManager(const char * path = "/tmp/odin0000.socket") : path_(path){
      unlink(path_.c_str());
      server_socket = socket(AF_LOCAL, SOCK_SEQPACKET, 0);
      if(server_socket == -1){
        Log(ERROR,"Unable to create server file descriptor" , strerror(errno));
      }
      sockaddr_un server_addr;
      memset(&server_addr, 0, sizeof(server_addr));
      server_addr.sun_family = AF_LOCAL;
      strncpy(server_addr.sun_path, path_.c_str(), sizeof(server_addr.sun_path) - 1);


      auto ret = bind(server_socket, reinterpret_cast<struct sockaddr*> (&server_addr), sizeof(server_addr));
      if(ret == -1){
        Log(ERROR, "Cannot binding server to addr" , strerror(errno));
      }
    }

    void start_listen(){
      auto ret = listen(server_socket,1);
      if(ret == -1 ){
        Log(ERROR, "Cannot listen to server fd" , server_socket);
      }
      client_handlers = std::thread(handle_client , server_socket , std::ref(prompt) , std::ref(stop_) , std::ref(condition_prompt_empty_), std::ref(prompt_mutex_));
    }

    std::optional<std::string> read_prompt(){
      std::unique_lock<std::mutex> lock(prompt_mutex_);
      condition_prompt_empty_.wait(lock , [this](){
          return !this->prompt.empty();
          });
      return prompt.pop();
    }


    void write_decode(const std::string& token){

      bool ret = false;
      {
        std::lock_guard<std::mutex> lock(decode_mutex_);
        ret = prompt.push(token) ;
      }
      if(!ret){
        Log(ERROR, "Push failed to decode" , strerror(errno));
      }else{
        condition_decode_empty_.notify_one();
      }

    }


    ~NetworkManager(){
      stop_.store(true);
      shutdown(server_socket, SHUT_RDWR);
      close(server_socket);

      if(client_handlers.joinable())
        client_handlers.join();

      unlink(path_.c_str());
    }
};
