#pragma once

#include "data_structures/lock_free_ring_buffer.hpp"
#include "data_structures/unidirectional_map.hpp"
#include "stream_buffer.hpp"
#include <condition_variable>
#include <csignal>
#include <netinet/in.h>
#include <string>
#include <sys/epoll.h>
#include <thread>
#include <vector>

enum class EventLoopAction{
  CONTINUE,
  BREAK,
};

struct MCSClient {
  stream_buffer buffer_;

  ringbuffer<std::string> requests;
  ringbuffer<std::string> response;

  int fd_;
  uint8_t fill_status;

  bool closed; 

  MCSClient(int fd) : buffer_(fd) , fd_(fd), fill_status(CLIENT_OPEN | DATA_NOT_PRESENT){ }
};

class MultiClientServer{
  private:
    static constexpr size_t MAX_CLIENTS = 100;

    const std::vector<const char *> file_paths = {
      "/index.html",
      "/style.css",
      "/dist/main.js",
    };

    unidirectional_map<std::string, std::string> file_content;

    std::thread handler_;

    std::atomic<bool> is_running_ = true;
    std::sig_atomic_t& interupt_;

    short port_;

    int server_fd_;
    int close_event_fd_;
    int infered_event_fd_;

    unidirectional_map<int, std::shared_ptr<MCSClient>> clients;

    sockaddr_in server_addr_;

    bool add_to_event(int epoll_fd , epoll_event& ev , int fd);
    EventLoopAction accept_clients(int epoll_fd ,epoll_event& ev );
    void handle_client_fill_state(int client_fd);
    void handle_event_close();
    void handle_client();

    std::mutex read_mutex_;
    std::condition_variable read_cv_;

  public:
    MultiClientServer(std::sig_atomic_t& interupt , short port); 
    void start_listen();
    void stop() ;
    std::string read_request() ;
    bool write_infered(const std::string& tok);

    ~MultiClientServer();
};
