#pragma once

#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

enum FillStatus : uint8_t{
  DATA_PRESENT = 1 << 0,
  DATA_NOT_PRESENT = 1 << 1,
  CLIENT_OPEN = 1 << 2,
  CLIENT_CLOSED = 1 << 3
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

    void compact() {
      if (read_head > 0) {
        std::memmove(
            data_.get(),
            data_.get() + read_head,
            size_ - read_head
            );

        size_ -= read_head;
        read_head = 0;
      }
    }

    void expand(){
      auto temp = std::make_unique<uint8_t[]>(capacity_*2); 
      std::copy_n(data_.get() , size_ , temp.get());

      capacity_ *= 2;

      data_ = std::move(temp);
    }

    void insert(uint8_t * begin , uint8_t * end){
      auto len = static_cast<size_t>(end - begin);
      if(size_ + len >= capacity_){
        compact();
      }
      while(size_ + len >= capacity_){
        expand();
      }
      std::copy(begin , end , data_.get() + size_);
      size_ += len;
    }

  public:
    stream_buffer(int fd) :
      size_(0),
      capacity_(init_capacity),
      data_(std::make_unique<uint8_t[]>(init_capacity)),
      read_head(0),
      read_fd_(fd)
  {}



    [[nodiscard]]
      size_t bytes_available(){
        size_t bytes_available = 0;
        ioctl(read_fd_, FIONREAD, &bytes_available);
        return bytes_available;
      }

    [[nodiscard]]
      bool is_readable(size_t size){
        return read_head + size <= size_;
      }


    [[nodiscard]]
      std::optional<uint8_t> read_u8(){
        if(read_head + sizeof(uint8_t) <= size_){

          auto val = data_[read_head];
          read_head ++;

          return val;
        }else{
          return std::nullopt;
        }
      }


    [[nodiscard]]
      std::optional<uint32_t> read_u32(){
        if(read_head + sizeof(uint32_t) <= size_){
          uint32_t val;

          memcpy(&val , data_.get() + read_head, sizeof(val));
          read_head += sizeof(val);

          return val;
        }else{
          return std::nullopt;
        }
      }

    void clear(int new_fd){
      read_fd_ = new_fd;
      read_head = 0;
      size_ = 0;
    }


    uint8_t fill() {
      uint8_t buffer[4096]; 
      bool data_read = false;

      while (true) {
        ssize_t bytes_read = recv(read_fd_, buffer, sizeof(buffer), MSG_DONTWAIT);
        if(bytes_read > 0){
          data_read = true;
          insert(std::begin(buffer), std::begin(buffer) + bytes_read);
        }else if (bytes_read == 0) {
          return data_read ? FillStatus::DATA_PRESENT | FillStatus::CLIENT_CLOSED : FillStatus::DATA_NOT_PRESENT | FillStatus::CLIENT_CLOSED; 
        }else if (bytes_read < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return data_read ? FillStatus::DATA_PRESENT | FillStatus::CLIENT_OPEN : FillStatus::DATA_NOT_PRESENT | FillStatus::CLIENT_OPEN; 
          }

          if (errno == EINTR) {
            continue;
          }

          if(errno == ECONNRESET){
            return FillStatus::DATA_NOT_PRESENT | FillStatus::CLIENT_CLOSED;
          }
        }
      }
    }

    decltype(auto) begin(){
      return data_.get() + read_head;
    }

    decltype(auto) end(){
      return data_.get() + size_;
    }

    std::optional<std::string> read_str(size_t s){
      if(is_readable(s)){
        std::string buf(begin() , begin() + s);
        read_head += s;
        return buf;
      }
      return std::nullopt;
    }

};
