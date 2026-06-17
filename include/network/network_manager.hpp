#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include "data_structures/lock_free_ring_buffer.hpp"
#include "logging.hpp"


struct Client{
  int fd_; 

  Client(){
    fd_ = -1;
  }
};
class NetworkManager {
  private:
    std::string path_;
    int server_socket_;


    ringbuffer<uint32_t> prompt;
    ringbuffer<uint32_t> infered;

    std::thread handler_thread;

  private:
    void handle_client(){
      Client client;

      int epoll_fd = epoll_create1(0);

      if (epoll_fd == -1) {
        Log(ERROR , "Failed to create epoll file descriptor\n");
        return;
      }

      if (close(epoll_fd)) {
        Log(ERROR, "Failed to close epoll file descriptor\n");
        return;
      }

      while(true){
        if(client.fd_ == -1){
          client.fd_ = accept(server_socket_, NULL, NULL);

          epoll_event event;
  

          epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client.fd_, struct epoll_event *event);
        }
      }
    }

  public:
    NetworkManager(const std::string& path = "/tmp/odin0000.socket") : path_(path){
      unlink(path_.c_str());
      server_socket_ = socket(AF_LOCAL, SOCK_STREAM, 0);
      if(server_socket_ == -1){
        Log(ERROR,"Unable to create server file descriptor" , strerror(errno));
      }
      sockaddr_un server_addr;
      memset(&server_addr, 0, sizeof(server_addr));
      server_addr.sun_family = AF_LOCAL;
      strncpy(server_addr.sun_path, path_.c_str(), sizeof(server_addr.sun_path) - 1);


      auto ret = bind(server_socket_, reinterpret_cast<struct sockaddr*> (&server_addr), sizeof(server_addr));
      if(ret == -1){
        Log(ERROR, "Cannot binding server to addr" , strerror(errno));
      }
    }

    void start_listen(){
      auto ret = listen(server_socket_, 1);
      if(ret == -1 ){
        Log(ERROR, "Cannot listen to server fd" , server_socket_);
      }
      handler_thread = std::thread(&NetworkManager::handle_client , this);
    }
};
