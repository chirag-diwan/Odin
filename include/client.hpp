#pragma once

#include <cstdint>
#include <netinet/in.h>
#include "stream_buffer.hpp"

enum class ClientState : uint8_t {
  READING_PAYLOAD,
  IDLE
};

struct Client{
  stream_buffer buffer_;
  int fd_;
  uint32_t len_;
  ClientState state_;
  uint8_t fill_status_;
  sockaddr_in client_addr_;

  Client(int fd) : buffer_(fd) , fd_(fd) , state_(ClientState::IDLE), fill_status_(CLIENT_OPEN | DATA_NOT_PRESENT){ }
}; 
