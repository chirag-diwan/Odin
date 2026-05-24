#include <cstdint>
#include <cstdio>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <vector>


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

    uint8_t* begin(){
      return buf.data() + r;
    }


    uint8_t* end(){
      return buf.data() + w;
    }
};
