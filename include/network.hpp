#include "logging.hpp"
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>

enum MsgType : uint8_t {
  TOKENS_IN = 1,
  TOKEN_OUT = 2,
  END_STREAM = 3,
  ERROR_MSG = 4,
};


class StreamBuffer {
  public:
    std::vector<uint8_t> buf;
    size_t r = 0; 
    size_t w = 0; 

    StreamBuffer(size_t cap) : buf(cap) {}

    size_t readable() const { return w - r; }
    size_t free_space() const { return buf.size() - w; }

    void compact() {
      if (r == 0) return;
      size_t n = readable();
      memmove(buf.data(), buf.data() + r, n);
      r = 0;
      w = n;
    }

    bool fill(int fd) {
      if (free_space() == 0) compact();

      ssize_t n = recv(fd, buf.data() + w, free_space(), 0);
      if (n <= 0) return false;

      w += n;
      return true;
    }

    bool read_u32(uint32_t &out) {
      if (readable() < sizeof(uint32_t)) return false;

      uint32_t net;
      memcpy(&net, buf.data() + r, sizeof(uint32_t));
      r += sizeof(uint32_t);

      out = ntohl(net);
      return true;
    }

    bool read_u8(uint8_t &out) {
      if (readable() < sizeof(uint8_t)) return false;

      uint8_t net;
      memcpy(&net, buf.data() + r, sizeof(uint8_t));
      r += sizeof(uint8_t);

      out = ntohl(net);
      return true;
    }
};


class Network{
  private:
    int server_fd;
    int port = 42069;
    std::condition_variable& token_vec_cv;
    std::mutex& token_vec_mutex;
    std::vector<uint64_t>& tokens;

  public:
    Network(std::vector<uint64_t>& tokens , std::condition_variable& token_vector_free_condition , std::mutex& token_vector_mutex)
      :token_vec_cv(token_vector_free_condition) ,
      token_vec_mutex(token_vector_mutex) ,
      tokens(tokens) {

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

    void StartListen(){
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

            std::cerr << token << "\n";
          }
        }
      }

      close(client_fd);
      close(server_fd);
    }
};
