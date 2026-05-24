#include "logging.hpp"
#include "streambuffer.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>

class NetworkManager{
  private:
    int server_fd;


  public:
    NetworkManager(){
      server_fd = socket(AF_INET, SOCK_STREAM, 0);
      if(server_fd == -1){
        Log(ERROR , "Cannot create server file descriptor");
        return;
      }

      sockaddr server_addr;
      server_addr.sa_family = AF_INET;
      inet_pton(AF_INET, "127.0.0.1", server_addr.sa_data);
      if(bind(server_fd, &server_addr, sizeof(server_addr)) < 0){
        Log(ERROR , "Cannot bind server fd with socker");
        return;
      }
    }

    void Start(){
      if(listen(server_fd, 1) < 0){
        Log(ERROR , "Cannot listen to server fd");
        return;
      }

      sockaddr client_addr;
      uint32_t client_addr_size = sizeof(client_addr);
      int client_fd = accept(server_fd,&client_addr,&client_addr_size);
      while(true){
        StreamBuffer sb(2048);
        sb.fill(client_fd);
        while(sb.readable() < sizeof(uint32_t))
          sb.fill(client_fd);

        uint32_t len;
        sb.read_u32(len);
        while(sb.readable() < len)
          sb.fill(client_fd);

        std::string json(sb.begin() , sb.end());
      }
    }
};
