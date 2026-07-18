#pragma once

#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>

enum FillStatus : uint8_t{
  DATA_PRESENT = 1 << 0,
  DATA_NOT_PRESENT = 1 << 1,
  CLIENT_OPEN = 1 << 2,
  CLIENT_CLOSED = 1 << 3,
  INTERUPT = 1 << 4
};

class stream_buffer{
  private:
    static constexpr size_t min_bytes_read = 256;
    static constexpr size_t init_capacity = 4096;
    uint32_t size_;
    uint32_t capacity_;
    std::unique_ptr<uint8_t[]> data_;

    size_t read_head;
    int read_fd_;

    void compact() ;
    void expand();

    void insert(uint8_t * begin , uint8_t * end);

  public:
    stream_buffer(int fd) :
      size_(0),
      capacity_(init_capacity),
      data_(std::make_unique<uint8_t[]>(init_capacity)),
      read_head(0),
      read_fd_(fd)
  {}

    [[nodiscard]]
      size_t bytes_available();

    [[nodiscard]]
      bool is_readable(size_t size);


    [[nodiscard]]
      std::optional<uint8_t> read_u8();


    [[nodiscard]]
      std::optional<uint32_t> read_u32();

    [[nodiscard]]
      bool cmp_last_few(const char *val) ;

    void clear(int new_fd);
    uint8_t fill() ;
    decltype(auto) begin();
    decltype(auto) end();
    std::optional<std::string> read_all_as_str();
    std::optional<std::string> read_str(size_t s);

};
