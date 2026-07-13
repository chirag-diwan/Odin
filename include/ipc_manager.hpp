#pragma once

#include <sys/socket.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <sys/un.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <condition_variable>
#include <csignal>
#include <mutex>
#include <string>
#include <thread>
#include "./data_structures/lock_free_ring_buffer.hpp"

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

    bool add_to_event(int epoll_fd , epoll_event& ev , int fd);

    void handle_client();

  public:

    IPCManager(std::sig_atomic_t& interupt , const std::string& path) ;

    void start_listen();

    std::string read_prompt() ;

    bool write_infered(const std::string& tok);

    void stop() ;

    ~IPCManager();
};
