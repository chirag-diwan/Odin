#pragma once

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include "../logging.hpp"
#include "../data_structures/lock_free_ring_buffer.hpp"
#include "../stream_buffer.hpp"


enum class ClientState : uint8_t{
  READING_LENGTH,
  READING_PAYLOAD,
};

struct Client{
  stream_buffer buffer_;
  int fd_;
  uint32_t len;
  ClientState state_;
  uint8_t fill_status;

  Client(int fd) : buffer_(fd) , fd_(fd) , state_(ClientState::READING_LENGTH){ }
}; 

class IPCManager {
  private:
    std::string path_;
    int server_socket;

    std::thread handler;

    ringbuffer<std::string> prompts;
    ringbuffer<std::string> infered;
    std::condition_variable prompt_not_empty_cv;
    std::mutex prompt_mutex;

    void handle_client(){
      int client_fd = -1;
      Client client(-1);

      while(true){
        if(client_fd == -1){
          client_fd = accept(server_socket, NULL, NULL);
          if(client_fd < 0){
            Log(ERROR , "Unable to accept client for" , server_socket);
            return;
          }
          client.fd_ = client_fd;
          client.buffer_.clear(client_fd);
        }

        client.fill_status = client.buffer_.fill();

        if(client.fill_status & DATA_PRESENT){
          while (true) {
            if (client.state_ == ClientState::READING_LENGTH) {
              if (!client.buffer_.is_readable(sizeof(uint32_t)))
                break;

              auto len = client.buffer_.read_u32();
              client.len = *len;
              client.state_ = ClientState::READING_PAYLOAD;
            }

            if (client.state_ == ClientState::READING_PAYLOAD) {
              if (!client.buffer_.is_readable(client.len))
                break;

              auto prompt = client.buffer_.read_str(client.len);

              auto ret = prompts.push(*prompt);
              if(ret){
                prompt_not_empty_cv.notify_one();
              }

              client.state_ = ClientState::READING_LENGTH;
            }
          }
        }

        if(!(client.fill_status & CLIENT_CLOSED)){
          if(!infered.empty()){
            std::string test_msg = *infered.pop();

            uint32_t total_bytes_sent = 0;
            uint32_t len = static_cast<uint32_t>(test_msg.size());

            while(total_bytes_sent < sizeof(len)){
              auto ret = send(client_fd, &len, sizeof(len), MSG_NOSIGNAL);
              if(ret > 0)total_bytes_sent += ret;
            }

            total_bytes_sent = 0;
            while (total_bytes_sent < test_msg.size()) {
              auto ret = send(client_fd, test_msg.c_str(), test_msg.size(), MSG_NOSIGNAL);
              if(ret > 0)total_bytes_sent += ret;
            }
          }
        }

        if(client.fill_status & CLIENT_CLOSED){
          close(client.fd_);
        }

        if((client.fill_status & CLIENT_CLOSED) &&
            (client.fill_status & DATA_NOT_PRESENT)){
          break;
        }
      }
      close(client.fd_);
    }

  public:

    IPCManager(const std::string& path = "/tmp/odin0000.socket") : path_(path) {
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
      auto ret = listen(server_socket, 1);
      if(ret < 0){
        Log(ERROR , "Listen failed for" , server_socket);
        return;
      }
      handler = std::thread(&IPCManager::handle_client , this );
    }


    std::string read_prompt(){
      std::unique_lock<std::mutex> lock(prompt_mutex);
      prompt_not_empty_cv.wait(lock , [this]()->bool{
          return !prompts.empty();
          });
      return *prompts.pop();
    }

    bool write_infered(const std::string& tok){
      return infered.push(tok);
    }

    ~IPCManager(){
      if(handler.joinable())handler.join();
      close(server_socket);
      unlink(path_.c_str());
    }
};
