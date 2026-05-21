#include "logging.hpp"
#include "chanel.hpp"
#include "streambuffer.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

enum MsgType : uint8_t {
  TOKENS_IN = 1,
  TOKEN_OUT = 2,
  END_STREAM = 3,
  ERROR_MSG = 4,
};


class Network{
  private:
    int server_fd;
    int port = 42069;

    size_t prev_token_count;
    std::vector<uint64_t>& tokens;
    std::thread network_thread;


    void startListen(Chanel& chanel){
      int res;
      res = listen(server_fd, 1);
      if (res < 0) {
        Log(ERROR , strerror(errno));
        close(server_fd);
        std::exit(-1);
      }

      Log(INFO , "Server listening on port" , port , "\n");


      sockaddr_in client_addr{};
      socklen_t client_len = sizeof(client_addr);

      int client_fd;
      client_fd = accept(server_fd,
          (sockaddr*)&client_addr,
          &client_len);

      if (client_fd < 0) {
        Log(ERROR , "accept failed");
        close(server_fd);
        std::exit(-1);
      }

      while(true){
        StreamBuffer sb(4096);

        while (sb.readable() < sizeof(uint32_t))
          if (!sb.fill(client_fd)) return;

        uint32_t length;
        sb.read_u32(length);

        while (sb.readable() < sizeof(uint8_t))
          if (!sb.fill(client_fd)) return;

        uint8_t type;
        sb.read_u8(type);

        if (type == MsgType::TOKENS_IN) {
          while (sb.readable() < sizeof(uint32_t))
            if (!sb.fill(client_fd)) return;

          uint32_t count;
          sb.read_u32(count);


          for (uint32_t i = 0; i < count; i++) {
            while (sb.readable() < sizeof(uint32_t))
              if (!sb.fill(client_fd)) return;

            uint32_t token;
            sb.read_u32(token);
            tokens.emplace_back(token);
          }
          prev_token_count = tokens.size();

        }else if(type == MsgType::TOKEN_OUT){
          uint64_t token = tokens[prev_token_count + 1];
          prev_token_count++;
          int bytes = send(client_fd, &token, sizeof(token), 0);
          if(bytes < 0){
            Log(ERROR , "error sending bytes" , strerror(errno));
            std::exit(-1);
          }
        }
      }
      close(client_fd);
    }

  public:
    Network(std::vector<uint64_t>& tokens) : tokens(tokens){
      prev_token_count = 0;

      port = 42069;

      server_fd = socket(AF_INET, SOCK_STREAM, 0);
      if(server_fd == -1){
        Log(ERROR , "Cannot open socket");
        std::exit(-1);
      }
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
      addr.sin_port = htons(port);

      int res;

      res = bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
      if (res < 0){
        Log(ERROR , strerror(errno));
        close(server_fd);
        std::exit(-1);
      }
    }

    void Start(){
      //TODO create a new thread that handles the network
    }


    ~Network(){
      close(server_fd);
    }
};
