#include "../../include/logging.hpp"
#include "../../include/experimental/multiclient-httpmanager.hpp"
#include "../../include/errors.hpp"
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/fcntl.h>
#include <format>
#include <unistd.h>



bool MultiClientServer::add_to_event(int epoll_fd , epoll_event& ev , int fd){
  ev.events = EPOLLIN;
  ev.data.fd = fd;

  if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) == -1) {
    Log(ERROR , "epoll_ctl failed for", fd);
    return false;
  }
  return true;
}

EventLoopAction MultiClientServer::accept_clients(int epoll_fd ,epoll_event& ev ){
  int client_fd = accept(server_fd_, NULL, NULL);
  if(client_fd < 0){
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return EventLoopAction::BREAK;     

    Log(ERROR, "accept failed", strerror(errno));
    return EventLoopAction::BREAK;
  }

  int flags = fcntl(client_fd, F_GETFL, 0);
  fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

  auto replaceable_client = clients.getValueOf(client_fd);

  if(!add_to_event(epoll_fd, ev, client_fd)){
    Log(ERROR, "Adding to event failed");
    close(client_fd);
    return EventLoopAction::CONTINUE;
  }

  if(replaceable_client && (*replaceable_client)->closed){
    (*replaceable_client) = std::make_shared<MCSClient>(client_fd);
  }else{
    clients.insert(client_fd , std::make_shared<MCSClient>(client_fd));
  }

  return EventLoopAction::CONTINUE;
}

void MultiClientServer::handle_client_fill_state(int client_fd){
  auto client = *clients.getValueOf(client_fd);

  if((client->fill_status & CLIENT_OPEN )){
    client->fill_status = client->buffer_.fill();
  } 

  if(client->fill_status & DATA_PRESENT){
    if (!client->buffer_.cmp_last_few("\r\n\r\n")){
      if(client->fill_status & CLIENT_CLOSED) {
        client->fill_status = CLIENT_CLOSED | DATA_NOT_PRESENT;
      }
      return;
    }

    auto prompt = client->buffer_.read_all_as_str();
    auto ret = client->requests.push(*prompt);

    if(ret){
      read_cv_.notify_one();
    }

    if(client->fill_status & CLIENT_CLOSED){
      client->fill_status = CLIENT_CLOSED | DATA_NOT_PRESENT;
    }else if (client->fill_status & CLIENT_OPEN){
      client->fill_status = CLIENT_OPEN | DATA_NOT_PRESENT;
    }
  }  else if(client->fill_status & CLIENT_CLOSED){
    client->closed = true;
    close(client->fd_);
  }
  return;
}

void MultiClientServer::handle_event_close(){
  uint64_t buf;
  read(close_event_fd_, &buf, sizeof(buf));

  for(auto& client : clients){
    if(client.occupied && !client.value->closed){
      close(client.key);
    }
  }

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

void MultiClientServer::handle_client(){
  epoll_event ev;

  int epoll_fd = epoll_create1(0);
  if(epoll_fd <= 0 ){
    Log(ERROR , "Cannot initialize epoll instance");
    return;
  }

  add_to_event(epoll_fd, ev, server_fd_);
  add_to_event(epoll_fd, ev, close_event_fd_);
  add_to_event(epoll_fd, ev, infered_event_fd_);

  epoll_event events[MAX_CLIENTS] = {};

  while(true){
    int nfds = epoll_wait(epoll_fd, events, MAX_CLIENTS, -1);

    if (nfds == -1) {
      Log(ERROR , "epoll_wait failed" , strerror(errno));
      break;
    }

    for(int i = 0 ; i < nfds ; i++){
      if( events[i].data.fd == server_fd_){
        while(true){
          if(accept_clients(epoll_fd, ev) == EventLoopAction::BREAK){
            break;
          }
        }
      }else if(events[i].data.fd == close_event_fd_){
        handle_event_close();
        close(epoll_fd);
        return;
      }else if(events[i].data.fd == infered_event_fd_){

        Errorif(true, "NOT IMPLEMENTED");

      }else{
        handle_client_fill_state(events[i].data.fd);
      }
    } 
  }

  for(auto& client : clients){
    if(client.occupied && !client.value->closed){
      close(client.key);
    }
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

MultiClientServer::MultiClientServer(std::sig_atomic_t& interupt , short port) :is_running_(true) , interupt_(interupt) , port_(port){
  clients.populate(MAX_CLIENTS);
  server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if(server_fd_ == -1){
    Log(ERROR,"Unable to create server file descriptor" , strerror(errno));
    return;
  }

  int flags = fcntl(server_fd_, F_GETFL, 0);
  fcntl(server_fd_, F_SETFL, flags | O_NONBLOCK);

  sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));

  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port_);

  if(inet_aton("127.0.0.1", &server_addr.sin_addr) == -1){
    Log(ERROR, "inet_aton failed" , strerror(errno));
  }

  int opt = 1;
  setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  auto ret = bind(server_fd_, reinterpret_cast<struct sockaddr*> (&server_addr), sizeof(server_addr));
  if(ret == -1){
    Log(ERROR, "Cannot binding server to addr" , strerror(errno));
  }

  close_event_fd_ = eventfd(0 , EFD_SEMAPHORE); //Binary semaphore
  infered_event_fd_ = eventfd(0 , EFD_SEMAPHORE); //Binary semaphore
}

void MultiClientServer::start_listen(){
  auto ret = listen(server_fd_, 100);

  if(ret < 0){
    Log(ERROR , "Listen failed for" , server_fd_);
    return;
  }

  Log(INFO,std::format("Listening on http://localhost:{}" , port_));
  Log(INFO,std::format("Max accepted client {}" , MAX_CLIENTS));
  handler_ = std::thread(&MultiClientServer::handle_client , this );
}

std::string MultiClientServer::read_request() {
  while(true){
    {
      std::unique_lock<std::mutex> _ (read_mutex_);
      read_cv_.wait_for(_, std::chrono::milliseconds(500),[this]{
        return interupt_ || !is_running_;
      });
    }

    if(interupt_){
      break;
    }

    if(!is_running_){
      break;
    }

    for(const auto& client : clients){
      if(client.occupied){
        if(!client.value->closed && !client.value->requests.empty()){
          return *client.value->requests.pop();
        }
      }
    }
  }
  return {};
}


bool MultiClientServer::write_infered(const std::string& tok){
  //auto ok = infered_.push(tok);

  //if(ok){
  //  uint64_t ret = 1;
  //  write(infered_event_fd_, &ret, sizeof(ret));
  //}

  //return ok;
  return false;
}


void MultiClientServer::stop() {
  uint64_t ret = 1;
  write(close_event_fd_, &ret, sizeof(ret));
  is_running_ = false;
  read_cv_.notify_all();
}

MultiClientServer::~MultiClientServer(){
  if (is_running_) {
    stop();
  }
  if (handler_.joinable()) {
    handler_.join();
  }
}
