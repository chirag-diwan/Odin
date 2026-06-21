#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

static bool send_all(int fd, const void* data, size_t len) {
  const char* ptr = static_cast<const char*>(data);
  size_t sent = 0;

  while (sent < len) {
    ssize_t ret = send(fd, ptr + sent, len - sent, MSG_NOSIGNAL);

    if (ret <= 0)
      return false;

    sent += ret;
  }

  return true;
}

static bool recv_all(int fd, void* data, size_t len) {
  char* ptr = static_cast<char*>(data);
  size_t received = 0;

  while (received < len) {
    ssize_t ret = recv(fd, ptr + received, len - received, 0);

    if (ret <= 0)
      return false;

    received += ret;
  }

  return true;
}

int main() {
  constexpr const char* SOCKET_PATH = "/tmp/odin0000.socket";

  int fd = socket(AF_LOCAL, SOCK_STREAM, 0);
  if (fd < 0) {
    perror("socket");
    return 1;
  }

  sockaddr_un addr{};
  addr.sun_family = AF_LOCAL;
  std::strncpy(addr.sun_path,
      SOCKET_PATH,
      sizeof(addr.sun_path) - 1);

  if (connect(
        fd,
        reinterpret_cast<sockaddr*>(&addr),
        sizeof(addr)) < 0) {
    perror("connect");
    close(fd);
    return 1;
  }

  std::atomic<bool> running{true};

  std::thread receiver([&]() {
      while (running) {
      uint32_t len = 0;

      if (!recv_all(fd, &len, sizeof(len))) {
      std::cout << "\nServer disconnected\n";
      running = false;
      break;
      }

      std::string msg(len, '\0');

      if (!recv_all(fd, msg.data(), len)) {
      std::cout << "\nServer disconnected\n";
      running = false;
      break;
      }

      std::cout << msg ;
      std::cout << std::flush;
      }
  });

  std::string line;

  while (running) {
    if (!std::getline(std::cin, line))
      break;

    line = "<|im_begin|>User<|im_end|><|im_begin|>Write a python program to write a file , and write \"I am a dumb AI\" in it.<|im_end|><|im_begin|> Assistant\n";

    uint32_t len = static_cast<uint32_t>(line.size());

    if (!send_all(fd, &len, sizeof(len))) {
      std::cout << "Failed to send length\n";
      break;
    }

    if (!send_all(fd, line.data(), len)) {
      std::cout << "Failed to send payload\n";
      break;
    }
  }

  running = false;
  shutdown(fd, SHUT_RDWR);

  if (receiver.joinable())
    receiver.join();

  close(fd);

  return 0;
}
