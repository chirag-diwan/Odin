#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include "../data_structures/lock_free_ring_buffer.hpp"
#include "../logging.hpp"
#include "./stream_buffer.hpp"

struct Client{
  stream_buffer buffer_;
  int socket_;
  uint32_t len_;
  uint32_t id;
  Client(int fd) : buffer_(fd) , socket_(fd) , len_(UINT32_MAX){ }
};

class NetworkManager{
  private:
    std::vector<Client> client;
  private:
    void handle_client(){
      int max_fd = 0;
      while(true){
        if(stop_)break;

        fd_set read_set; FD_ZERO(&read_set);
        FD_SET(server_socket, &read_set);

        max_fd = server_socket;

        for (size_t i = 0; i < max_clients; i++) {
          if(i >= client.size()) break;
          int sd = client[i].socket_;
          if (sd >= 0) {
            FD_SET(sd, &read_set);
          }
          if (sd > max_fd) {
            max_fd = sd; 
          }
        }

        int activity = select(max_fd + 1, &read_set, NULL, NULL, NULL);
        if (activity < 0) {
          if (errno == EINTR)
            continue;
          Log(ERROR , "Select error" , strerror(errno));
        }

        if (FD_ISSET(server_socket, &read_set)) {
          int new_socket = accept(server_socket, NULL, NULL);
          if ((new_socket ) < 0) {
            Log(ERROR , "Accept error" , strerror(errno));
            exit(EXIT_FAILURE);
          }

          if (client.size() >= max_clients) {
            close(new_socket);
            continue;
          }

          uint32_t client_id = client.size();

          send(new_socket, &client_id, sizeof(client_id), 0);

          for (size_t i = 0; i < max_clients; i++) {
            if(client.size() < max_clients){
              client.emplace_back(Client(new_socket));
              break;
            }
            if (client[i].socket_ == 0) {
              client[i] = Client(new_socket);
              break;
            }
          }
        }

        for (size_t i = 0 ; i < client.size() ; i++) {
          auto& c = client[i];
          int sd = c.socket_;

          if (FD_ISSET(sd, &read_set)) {
            stream_buffer& buf = c.buffer_;

            if(c.len_ == UINT32_MAX){

              if(buf.is_readable(sizeof(uint32_t))){
                auto len = buf.read_u32();
                if(len){
                  c.len_ = *len;
                }else{
                  Log(ERROR , "buf.read_u32 failed" , strerror(errno));
                }
              }else{
                if(!buf.fill(sizeof(uint32_t))){
                  close(c.socket_);
                  c.socket_ = 0;
                }
              }
            }else{
              if(!buf.is_readable(c.len_)){
                if(!buf.fill()){
                  close(c.socket_);
                  c.socket_ = -1;
                  c.len_ = UINT32_MAX;
                }
              }else{
                auto temp = buf.read_str(c.len_);
                if(!temp.has_value())continue;

                auto ret = prompts.push(*temp);
                c.len_ = UINT32_MAX;
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
    }


    std::mutex prompt_mutex_;
    std::condition_variable condition_prompt_empty_;

    const std::string path_;

    ringbuffer<std::string> prompts;
    ringbuffer<std::string> infered;

    std::thread client_handlers;

    std::atomic<bool> stop_ = false;
  public:
    static constexpr size_t max_clients = 30;
    static constexpr char client_base_name[] = "client_no_";

    int server_socket = -1;
    NetworkManager(const char * path = "/tmp/odin0000.socket") : path_(path) {
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
          return stop_ || !this->prompts.empty();
          });
      return prompts.pop();
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
