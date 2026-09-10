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

    std::condition_variable readCv_;
    std::mutex promptMutex_;


    std::atomic<bool> isRunning_;
    std::shared_ptr<std::sig_atomic_t> interupt_;


    int serverFd_;
    int closeEventFd_;
    int inferedEventFd_;

    bool addToEvent(int epoll_fd , epoll_event& ev , int fd);

    void handleClient();

  public:
    void Init(std::shared_ptr<std::sig_atomic_t> interupt , const std::string& path = "/tmp/odin0000.socket") ;

    void StartListen();

    std::string ReadPrompt() ;

    bool WriteInfered(const std::string& tok);

    void Stop() ;

    void Delete();
};
