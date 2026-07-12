#pragma once

#include <sys/socket.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <sys/un.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <condition_variable>
#include <csignal>
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

  Client(int fd) : buffer_(fd) , fd_(fd) , state_(ClientState::READING_LENGTH), fill_status(CLIENT_CLOSED | DATA_NOT_PRESENT){ }
}; 

class IPCManager {
  private:
    std::string path_;

    std::thread handler_;

    ringbuffer<std::string> prompts_;
    ringbuffer<std::string> infered_;

    std::condition_variable read_cv_;
    std::mutex prompt_mutex;


    std::atomic<bool> is_running_;
    std::sig_atomic_t& interupt_;


    int server_fd_;
    int close_event_fd_;
    int infered_event_fd_;

    bool add_to_event(int epoll_fd , epoll_event& ev , int fd){
      ev.events = EPOLLIN;
      ev.data.fd = fd;

      if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) == -1) {
        Log(ERROR , "epoll_ctl failed for",fd);
        return false;
      }
      return true;
    }

    void handle_client(){
      constexpr size_t MAX_EVENT = 10;
      epoll_event ev;

      int epoll_fd = epoll_create1(0);
      if(epoll_fd <= 0 ){
        Log(ERROR , "Cannot initialize epoll instance");
        return;
      }

      add_to_event(epoll_fd, ev, server_fd_);
      add_to_event(epoll_fd, ev, close_event_fd_);
      add_to_event(epoll_fd, ev, infered_event_fd_);

      epoll_event events[MAX_EVENT];

      int client_fd = -1;
      Client client(client_fd);

      while(true){
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENT, -1);
        if (nfds == -1) {
          Log("epoll_wait failed");
          continue;
        }

        for(int i = 0 ; i < nfds ; i++){
          if((client.fill_status & DATA_NOT_PRESENT ) && (client.fill_status & CLIENT_CLOSED) && events[i].data.fd == server_fd_){
            client_fd = accept(server_fd_, NULL, NULL);
            if(client_fd < 0){
              Log(ERROR , "Unable to accept client for" , server_fd_);
              continue;
            }

            int flags = fcntl(client_fd, F_GETFL, 0);
            fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

            client.fd_ = client_fd;
            client.buffer_.clear(client.fd_);
            client.fill_status = CLIENT_OPEN;

            if(!add_to_event(epoll_fd, ev, client_fd)){
              break;
            }
          }else if(events[i].data.fd == close_event_fd_){
            uint64_t buf;
            read(close_event_fd_, &buf, sizeof(buf));

            if(client.fd_ != -1){
              close(client.fd_);
            }

            close(epoll_fd);


            if (close_event_fd_ >= 0) {
              close(close_event_fd_);
              close_event_fd_ = -1;
            }

            if (server_fd_ >= 0) {
              shutdown(server_fd_, SHUT_RDWR);
              close(server_fd_);
              server_fd_ = -1;
            }

            return;
          }else if(events[i].data.fd == infered_event_fd_){
            uint64_t buf;
            read(infered_event_fd_, &buf, sizeof(buf));

            if(client.fill_status & CLIENT_OPEN){
              if(!infered_.empty()){
                std::string token = *infered_.pop();

                uint32_t total_bytes_sent = 0;
                uint32_t len = static_cast<uint32_t>(token.size());

                while(total_bytes_sent < sizeof(len)){
                  auto ret = send(client_fd, &len + total_bytes_sent, sizeof(len) - total_bytes_sent, MSG_NOSIGNAL);
                  if(ret > 0)total_bytes_sent += ret;
                }

                total_bytes_sent = 0;
                while (total_bytes_sent < token.size()) {
                  auto ret = send(client_fd, token.c_str() + total_bytes_sent, token.size() - total_bytes_sent, MSG_NOSIGNAL);

                  if (ret > 0) {
                    total_bytes_sent += ret;
                  } else if (ret == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                    break; 
                  } else {
                    client.fill_status = CLIENT_CLOSED;
                    break;
                  }
                }
              }
            }

          }else if((client.fill_status & CLIENT_OPEN ) && events[i].data.fd == client.fd_){ client.fill_status = client.buffer_.fill();
          }
        } 

        if(client.fill_status & DATA_PRESENT){
          while (true) {
            if (client.state_ == ClientState::READING_LENGTH) {
              if (!client.buffer_.is_readable(sizeof(uint32_t))){
                if(client.fill_status & CLIENT_CLOSED) {
                  client.fill_status = CLIENT_CLOSED | DATA_NOT_PRESENT;
                }
                break;
              }

              auto len = client.buffer_.read_u32();
              client.len = *len;
              client.state_ = ClientState::READING_PAYLOAD;
            }

            if (client.state_ == ClientState::READING_PAYLOAD) {
              if (!client.buffer_.is_readable(client.len)){
                if(client.fill_status & CLIENT_CLOSED) {
                  client.fill_status = CLIENT_CLOSED | DATA_NOT_PRESENT;
                }

                break;
              }

              auto prompt = client.buffer_.read_str(client.len);

              auto ret = prompts_.push(*prompt);
              if(ret){
                read_cv_.notify_one();
              }

              client.state_ = ClientState::READING_LENGTH;
            }
          }
        }


        if(client.fill_status & CLIENT_CLOSED){
          close(client.fd_);
          client.fd_ = -1;
        }
      }

      if(client.fd_ != -1){
        close(client.fd_);
      }

      close(epoll_fd);


      if (close_event_fd_ >= 0) {
        close(close_event_fd_);
        close_event_fd_ = -1;
      }

      if (server_fd_ >= 0) {
        shutdown(server_fd_, SHUT_RDWR);
        close(server_fd_);
        server_fd_ = -1;
      }
    }

  public:

    IPCManager(std::sig_atomic_t& interupt , const std::string& path = "/tmp/odin0000.socket") : path_(path)  , is_running_(true) , interupt_(interupt){
      unlink(path_.c_str());
      server_fd_ = socket(AF_LOCAL, SOCK_STREAM, 0);
      if(server_fd_ == -1){
        Log(ERROR,"Unable to create server file descriptor" , strerror(errno));
        return;
      }

      int flags = fcntl(server_fd_, F_GETFL, 0);
      fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);

      sockaddr_un server_addr;
      memset(&server_addr, 0, sizeof(server_addr));
      server_addr.sun_family = AF_LOCAL;
      strncpy(server_addr.sun_path, path_.c_str(), sizeof(server_addr.sun_path) - 1);
      auto ret = bind(server_fd_, reinterpret_cast<struct sockaddr*> (&server_addr), sizeof(server_addr));
      if(ret == -1){
        Log(ERROR, "Cannot binding server to addr" , strerror(errno));
      }

      close_event_fd_ = eventfd(0 , EFD_SEMAPHORE); //Binary semaphore
      infered_event_fd_ = eventfd(0 , EFD_SEMAPHORE); //Binary semaphore
    }

    void start_listen(){
      auto ret = listen(server_fd_, 1);
      if(ret < 0){
        Log(ERROR , "Listen failed for" , server_fd_);
        return;
      }


      handler_ = std::thread(&IPCManager::handle_client , this );
    }

    std::string read_prompt() {
      std::unique_lock<std::mutex> lock(prompt_mutex);
      while(true){

        bool got_data = read_cv_.wait_for(lock, std::chrono::milliseconds(500), [this] {
            return !is_running_ || !prompts_.empty();
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


    bool write_infered(const std::string& tok){
      auto ok = infered_.push(tok);

      if(ok){
        uint64_t ret = 1;
        write(infered_event_fd_, &ret, sizeof(ret));
      }

      return ok;
    }


    void stop() {
      uint64_t ret = 1;
      write(close_event_fd_, &ret, sizeof(ret));

      is_running_ = false;

      read_cv_.notify_all();
    }

    ~IPCManager(){
      if (is_running_) {
        stop();
      }
      if (handler_.joinable()) {
        handler_.join();
      }
      unlink(path_.c_str());
    }
};
