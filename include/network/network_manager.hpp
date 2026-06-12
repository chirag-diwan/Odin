#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include "../data_structures/lock_free_ring_buffer.hpp"
#include "../logging.hpp"
#include "./stream_buffer.hpp"


class NetworkManager{
  private:
    static constexpr size_t max_clients = 30;
    std::vector<int> client_sockets;
  private:
    void handle_client(){
      int max_fd = 0;
      while(true){
        if(stop_)break;

        fd_set read_set; FD_ZERO(&read_set);
        FD_SET(server_socket, &read_set);
        max_fd = server_socket;

        for (size_t i = 0; i < max_clients; i++) {
          int sd = client_sockets[i];
          if (sd > 0) {
            FD_SET(sd, &read_set);
          }
          if (sd > max_fd) {
            max_fd = sd; 
          }
        }

        int activity = select(max_fd + 1, &read_set, NULL, NULL, NULL);
        if (activity < 0) {
          perror("Select error");
        }

        if (FD_ISSET(server_socket, &read_set)) {
          int new_socket = accept(server_socket, NULL, NULL);
          if ((new_socket ) < 0) {
            perror("Accept error");
            exit(EXIT_FAILURE);
          }

          for (size_t i = 0; i < max_clients; i++) {
            if (client_sockets[i] == 0) {
              client_sockets[i] = new_socket;
              break;
            }
          }
        }

        for (size_t i = 0; i < max_clients; i++) {
          int sd = client_sockets[i];

          if (FD_ISSET(sd, &read_set)) {
            stream_buffer buf(sd);

            while(!buf.is_readable(sizeof(uint32_t))){
              if(!buf.fill(sizeof(uint32_t)))break;
            }

            auto len = buf.read_u32();
            if(!len.has_value())continue;

            while(!buf.is_readable(*len)){
              if(!buf.fill(*len)) {
                close(sd);
                client_sockets[i] = 0;
              }
            }

            std::string prompt_str;
            prompt_str.insert(prompt_str.end() , buf.begin() , buf.end());

            if(!prompt_str.empty()){
              bool ret = false;
              {
                std::lock_guard<std::mutex> lock(prompt_mutex_);
                ret = prompt.push(prompt_str) ;
              }
              if(!ret){
                Log(ERROR, "Push failed to prompt" , strerror(errno));
              }else{
                condition_prompt_empty_.notify_one();
              }
            }
          }
        }
      }
    }


    std::mutex prompt_mutex_;
    std::condition_variable condition_prompt_empty_;

    const std::string path_;

    ringbuffer<std::string> prompt;
    ringbuffer<std::string> infered;

    std::thread client_handlers;

    std::atomic<bool> stop_ = false;
  public:

    int server_socket = -1;
    NetworkManager(const char * path = "/tmp/odin0000.socket") :client_sockets(max_clients), path_(path) {
      unlink(path_.c_str());
      server_socket = socket(AF_LOCAL, SOCK_STREAM, 0);
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
      auto ret = listen(server_socket,5);
      if(ret == -1 ){
        Log(ERROR, "Cannot listen to server fd" , server_socket);
      }
      client_handlers = std::thread(&NetworkManager::handle_client , this);
    }

    std::optional<std::string> read_prompt(){
      std::unique_lock<std::mutex> lock(prompt_mutex_);
      condition_prompt_empty_.wait(lock , [this](){
          return stop_ || !this->prompt.empty();
          });
      return prompt.pop();
    }

    ~NetworkManager(){
      stop_.store(true);
      condition_prompt_empty_.notify_all();
      shutdown(server_socket, SHUT_RDWR);
      close(server_socket);

      if(client_handlers.joinable())
        client_handlers.join();

      unlink(path_.c_str());
    }
};
