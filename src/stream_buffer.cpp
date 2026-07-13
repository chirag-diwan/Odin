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

#include "../include/stream_buffer.hpp"

void stream_buffer::compact() {
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

void stream_buffer::expand(){
  auto temp = std::make_unique<uint8_t[]>(capacity_*2); 
  std::copy_n(data_.get() , size_ , temp.get());

  capacity_ *= 2;

  data_ = std::move(temp);
}

void stream_buffer::insert(uint8_t * begin , uint8_t * end){
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

[[nodiscard]]
size_t stream_buffer::bytes_available(){
  int bytes_available = 0;
  ioctl(read_fd_, FIONREAD, &bytes_available);
  return bytes_available;
}

[[nodiscard]]
bool stream_buffer::is_readable(size_t size){
  return read_head + size <= size_;
}


[[nodiscard]]
std::optional<uint8_t> stream_buffer::read_u8(){
  if(read_head + sizeof(uint8_t) <= size_){

    auto val = data_[read_head];
    read_head ++;

    return val;
  }else{
    return std::nullopt;
  }
}


[[nodiscard]]
std::optional<uint32_t> stream_buffer::read_u32(){
  if(read_head + sizeof(uint32_t) <= size_){
    uint32_t val;

    memcpy(&val , data_.get() + read_head, sizeof(val));
    read_head += sizeof(val);

    return val;
  }else{
    return std::nullopt;
  }
}

[[nodiscard]]
bool stream_buffer::cmp_last_few(const char *val) {
  size_t val_len  = strlen(val);

  if (val_len <= size_)
  {
    return memcmp(data_.get() + size_ - val_len, val, val_len) == 0;
  }

  return false;
}

void stream_buffer::clear(int new_fd){
  read_fd_ = new_fd;
  read_head = 0;
  size_ = 0;
}


uint8_t stream_buffer::fill() {
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
        if(data_read){
          return FillStatus::DATA_PRESENT | FillStatus::CLIENT_CLOSED;
        }
        return FillStatus::DATA_NOT_PRESENT | FillStatus::CLIENT_CLOSED;
      }
    }
  }
}

decltype(auto) stream_buffer::begin(){
  return data_.get() + read_head;
}

decltype(auto) stream_buffer::end(){
  return data_.get() + size_;
}


std::optional<std::string> stream_buffer::read_all_as_str(){
  if(size_ == 0){
    return std::nullopt;
  }
  std::string buf(begin() , end());
  read_head = size_;
  return buf;
}

std::optional<std::string> stream_buffer::read_str(size_t s){
  if(is_readable(s)){
    std::string buf(begin() , begin() + s);
    read_head += s;
    return buf;
  }
  return std::nullopt;
}
