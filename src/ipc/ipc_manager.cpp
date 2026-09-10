#include "../../include/ipc_manager.hpp"
#include "../../include/logging.hpp"
#include "../../include/client.hpp"


bool IPCManager::addToEvent(int epoll_fd , epoll_event& ev , int fd){
  ev.events = EPOLLIN;
  ev.data.fd = fd;

  if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &ev) == -1) {
    Log(ERROR , "epoll_ctl failed for",fd);
    return false;
  }
  Log(INFO, "Added fd to epoll:", fd);
  return true;
}

void IPCManager::handleClient(){
  constexpr size_t MAX_EVENT = 10;
  epoll_event ev;

  int epoll_fd = epoll_create1(0);
  if(epoll_fd <= 0 ){
    Log(ERROR , "Cannot initialize epoll instance");
    return;
  }

  Log(INFO, "IPC server event loop started");

  addToEvent(epoll_fd, ev, serverFd_);
  addToEvent(epoll_fd, ev, closeEventFd_);
  addToEvent(epoll_fd, ev, inferedEventFd_);

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
      if((client.fill_status_ & DATA_NOT_PRESENT ) && (client.fill_status_ & CLIENT_CLOSED) && events[i].data.fd == serverFd_){
        client_fd = accept(serverFd_, NULL, NULL);
        if(client_fd < 0){
          Log(ERROR , "Unable to accept client for" , serverFd_);
          continue;
        }

        Log(INFO, "Client connected:", client_fd);

        int flags = fcntl(client_fd, F_GETFL, 0);
        fcntl(client_fd, F_SETFL, flags | O_NONBLOCK);

        client.fd_ = client_fd;
        client.buffer_.clear(client.fd_);
        client.fill_status_ = CLIENT_OPEN;

        if(!addToEvent(epoll_fd, ev, client_fd)){
          break;
        }
      }else if(events[i].data.fd == closeEventFd_){
        uint64_t buf;
        read(closeEventFd_, &buf, sizeof(buf));

        Log(INFO, "IPC server shutdown requested");

        if(client.fd_ != -1){
          Log(INFO, "Closing client:", client.fd_);
          close(client.fd_);
        }

        close(epoll_fd);


        if (closeEventFd_ >= 0) {
          close(closeEventFd_);
          closeEventFd_ = -1;
        }

        if (serverFd_ >= 0) {
          Log(INFO, "Closing server socket:", serverFd_);
          shutdown(serverFd_, SHUT_RDWR);
          close(serverFd_);
          serverFd_ = -1;
        }

        Log(INFO, "IPC server stopped");
        return;
      }else if(events[i].data.fd == inferedEventFd_){
        uint64_t buf;
        read(inferedEventFd_, &buf, sizeof(buf));

        if(client.fill_status_ & CLIENT_OPEN){
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
                Log(INFO, "Client disconnected while sending response:", client.fd_);
                client.fill_status_ = CLIENT_CLOSED;
                break;
              }
            }
          }
        }

      }else if((client.fill_status_ & CLIENT_OPEN ) && events[i].data.fd == client.fd_){
        client.fill_status_ = client.buffer_.fill();

        if(client.fill_status_ & CLIENT_CLOSED){
          Log(INFO, "Client closed connection:", client.fd_);
        }
      }
    } 

    if(client.fill_status_ & DATA_PRESENT){
      while (true) {
        if (client.state_ == ClientState::IDLE) {
          if (!client.buffer_.is_readable(sizeof(uint32_t))){
            if(client.fill_status_ & CLIENT_CLOSED) {
              client.fill_status_ = CLIENT_CLOSED | DATA_NOT_PRESENT;
            }
            break;
          }

          auto len = client.buffer_.read_u32();
          client.len_ = *len;
          client.state_ = ClientState::READING_PAYLOAD;

          Log(INFO, "Receiving request from client:", client.fd_, "payload size:", client.len_);
        }

        if (client.state_ == ClientState::READING_PAYLOAD) {
          if (!client.buffer_.is_readable(client.len_)){
            if(client.fill_status_ & CLIENT_CLOSED) {
              client.fill_status_ = CLIENT_CLOSED | DATA_NOT_PRESENT;
            }

            break;
          }

          auto prompt = client.buffer_.read_str(client.len_);

          auto ret = prompts_.push(*prompt);
          if(ret){
            readCv_.notify_one();
          }

          client.state_ = ClientState::IDLE;
        }
      }
    }


    if(client.fill_status_ & CLIENT_CLOSED){
      Log(INFO, "Closing client connection:", client.fd_);
      close(client.fd_);
      client.fd_ = -1;
    }
  }

  if(client.fd_ != -1){
    Log(INFO, "Closing client connection:", client.fd_);
    close(client.fd_);
  }

  close(epoll_fd);


  if (closeEventFd_ >= 0) {
    close(closeEventFd_);
    closeEventFd_ = -1;
  }

  if (serverFd_ >= 0) {
    Log(INFO, "Closing server socket:", serverFd_);
    shutdown(serverFd_, SHUT_RDWR);
    close(serverFd_);
    serverFd_ = -1;
  }

  Log(INFO, "IPC server event loop exited");
}


void IPCManager::Init(std::shared_ptr<std::sig_atomic_t> interupt , const std::string& path){
  path_ = (path);
  isRunning_ = (true);
  interupt_ = (interupt);
  unlink(path_.c_str());

  Log(INFO, "Initializing IPC server:", path_);

  serverFd_ = socket(AF_LOCAL, SOCK_STREAM, 0);
  if(serverFd_ == -1){
    Log(ERROR,"Unable to create server file descriptor" , strerror(errno));
    return;
  }

  Log(INFO, "Created server socket:", serverFd_);

  int flags = fcntl(serverFd_, F_GETFL, 0);
  fcntl(serverFd_, F_SETFL, flags | O_NONBLOCK);

  sockaddr_un server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sun_family = AF_LOCAL;
  strncpy(server_addr.sun_path, path_.c_str(), sizeof(server_addr.sun_path) - 1);
  auto ret = bind(serverFd_, reinterpret_cast<struct sockaddr*> (&server_addr), sizeof(server_addr));
  if(ret == -1){
    Log(ERROR, "Cannot binding server to addr" , strerror(errno));
  } else {
    Log(INFO, "IPC server bound to:", path_);
  }

  closeEventFd_ = eventfd(0 , EFD_SEMAPHORE); //Binary semaphore
  inferedEventFd_ = eventfd(0 , EFD_SEMAPHORE); //Binary semaphore

  Log(INFO, "IPC event descriptors initialized");
}

void IPCManager::StartListen(){
  auto ret = listen(serverFd_, 1);
  if(ret < 0){
    Log(ERROR , "Listen failed for" , serverFd_);
    return;
  }

  Log(INFO, "IPC server listening on:", path_);

  handler_ = std::thread(&IPCManager::handleClient , this );
}

std::string IPCManager::ReadPrompt() {
  std::unique_lock<std::mutex> lock(promptMutex_);
  while(true){

    bool got_data = readCv_.wait_for(lock, std::chrono::milliseconds(500), [this] {
      return !isRunning_ || !prompts_.empty();
    });

    if (!got_data) {
      if (*interupt_) {
        Log(INFO, "Interrupt received while waiting for prompt");
        return {}; 
      }
      continue;
    }else{
      break;
    }
  }

  if (prompts_.empty()) return {};

  auto prompt = *prompts_.pop();
  Log(INFO, "Prompt retrieved for inference");

  return prompt;
}


bool IPCManager::WriteInfered(const std::string& tok){
  auto ok = infered_.push(tok);

  if(ok){
    uint64_t ret = 1;
    write(inferedEventFd_, &ret, sizeof(ret));

  } else {
    Log(INFO, "Failed to queue inference result");
  }

  return ok;
}


void IPCManager::Stop() {
  Log(INFO, "Stopping IPC server");

  uint64_t ret = 1;
  write(closeEventFd_, &ret, sizeof(ret));

  isRunning_ = false;

  readCv_.notify_all();
}

void IPCManager::Delete(){
  if (isRunning_) {
    Stop();
  }

  if (handler_.joinable()) {
    handler_.join();
  }

  unlink(path_.c_str());
}

