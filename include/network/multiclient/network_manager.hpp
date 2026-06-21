#pragma once

#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <atomic>
#include "../../data_structures/lock_free_ring_buffer.hpp"
#include "../../logging.hpp"
#include "../stream_buffer.hpp"

enum class ClientState : uint8_t{
  READING_LENGTH,
  READING_PAYLOAD
};

struct Client{
  stream_buffer buffer_;
  int socket_;
  uint32_t len_;
  uint32_t id_;
  ClientState state_;
  uint8_t fill_status_;
  Client(int fd , uint32_t id) : buffer_(fd) , socket_(fd) , len_(UINT32_MAX) , id_(id) , state_(ClientState::READING_LENGTH){ }
};

class NetworkManager{
  private:
    std::vector<Client> client;

  private:
    void handle_client(){
      int max_fd = 0;
      size_t total_client_count = 0;

      while(true){
        if(stop_)break;

        fd_set read_set; FD_ZERO(&read_set);
        FD_SET(server_socket, &read_set);

        max_fd = server_socket;

        for (size_t i = 0; i < client.size(); i++) {
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

          if (total_client_count >= max_clients) {
            close(new_socket);
            continue;
          }

          uint32_t client_id = total_client_count;
          send(new_socket, &client_id, sizeof(client_id), 0);

          for (size_t i = 0; i < max_clients; i++) {
            if(client.size() < max_clients){
              client.emplace_back(Client(new_socket , client_id));

              total_client_count++;
              break;
            }else if (client[i].socket_ == -1) {

              client[i].socket_ = new_socket;
              client[i].buffer_.clear(new_socket);
              client[i].id_ = client_id;

              total_client_count++;
              break;
            }
          }
        }

        for (size_t i = 0 ; i < client.size() ; i++) {
          auto& c = client[i];

          if(c.socket_ == -1) continue;

          if (FD_ISSET(c.socket_, &read_set)) {
            c.fill_status_ = c.buffer_.fill();

            if(c.fill_status_ & CLIENT_CLOSED){
              close(c.socket_);
              c.socket_ = -1;
              if(total_client_count > 0){
                total_client_count--;
              }
            }
          }
        }

        for(size_t i = 0 ; i < client.size() ; i++){
          auto& c = client[i];

          if(!(c.fill_status_&DATA_PRESENT))continue;

          stream_buffer& buf = c.buffer_;

          if(c.state_ == ClientState::READING_LENGTH){
            if(buf.is_readable(sizeof(uint32_t))){
              c.len_ = *buf.read_u32();
              c.state_ = ClientState::READING_PAYLOAD;
            }
          }

          if(c.state_ == ClientState::READING_PAYLOAD){
            if(buf.is_readable(c.len_)){
              c.state_ = ClientState::READING_LENGTH;

              if(!prompts.push(*buf.read_str(c.len_))){
                Log(ERROR, "Push failed to prompt" , strerror(errno));
              }else{
                push_sequence_.fetch_add(1, std::memory_order_release);
                push_sequence_.notify_one();
              }
            }
          }
        }


        for(size_t i = 0 ; i < client.size() ; i++){
          auto& c = client[i];

          if(infered.empty()){
            continue;
          } 
          if(c.fill_status_&CLIENT_CLOSED)continue;

          auto tokens = *infered.pop();

          size_t total_bytes_sent = 0;

          while (total_bytes_sent < tokens.size()) {
            ssize_t ret = send(
                c.socket_,
                tokens.data() + total_bytes_sent,
                tokens.size() - total_bytes_sent,
                MSG_NOSIGNAL
                );

            if (ret <= 0) {
              c.fill_status_ = CLIENT_CLOSED | DATA_NOT_PRESENT;
              break;
            }

            total_bytes_sent += ret;
          }
        }
      }
    }


    const std::string path_;

    ringbuffer<std::string> prompts;
    ringbuffer<std::string> infered;

    std::thread client_handlers;

    std::atomic<bool> stop_ = false;
    std::atomic<uint64_t> push_sequence_ = 0;

  public:
    static constexpr size_t max_clients = 30;

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
      while(true){
        if(stop_.load(std::memory_order_acquire)) return std::nullopt;
        if(!prompts.empty()){
          return prompts.pop();
        }

        uint64_t current_seq = push_sequence_.load(std::memory_order_acquire);

        if(!prompts.empty()) continue;
        push_sequence_.wait(current_seq , std::memory_order_acquire);
      }
      return prompts.pop();
    }

    bool write_infered(const std::string& token){
      if(token.size() == 0)return true;
      return infered.push(token);
    }

    ~NetworkManager(){
      stop_.store(true);
      push_sequence_.fetch_add(1, std::memory_order_release);
      push_sequence_.notify_all();
      shutdown(server_socket, SHUT_RDWR);
      close(server_socket);

      if(client_handlers.joinable())
        client_handlers.join();

      unlink(path_.c_str());
    }
};
