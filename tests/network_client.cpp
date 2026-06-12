#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <cerrno>

static bool send_all(int fd, const void* data, size_t len) {
  const char* ptr = static_cast<const char*>(data);

  while (len > 0) {
    ssize_t sent = send(fd, ptr, len, 0);

    if (sent < 0) {
      if (errno == EINTR) continue; // retry
      return false;
    }

    if (sent == 0) return false;

    ptr += sent;
    len -= static_cast<size_t>(sent);
  }
  return true;
}

class OdinClient {
  public:
    explicit OdinClient(const std::string& socket_path)
      : path_(socket_path) {}

    bool connect_to_server() {
      sock_ = socket(AF_UNIX, SOCK_STREAM, 0);
      if (sock_ < 0) {
        perror("socket");
        return false;
      }

      sockaddr_un addr{};
      std::memset(&addr, 0, sizeof(addr));

      addr.sun_family = AF_UNIX;
      std::strncpy(addr.sun_path, path_.c_str(), sizeof(addr.sun_path) - 1);

      size_t addr_len =
        offsetof(sockaddr_un, sun_path) + path_.size();

      if (connect(sock_, reinterpret_cast<sockaddr*>(&addr), addr_len) < 0) {
        perror("connect");
        close(sock_);
        sock_ = -1;
        return false;
      }

      return true;
    }

    bool send_message(const std::string& msg) {
      if (sock_ < 0) return false;

      uint32_t len = static_cast<uint32_t>(msg.size());
      uint32_t net_len = htonl(len);

      if (!send_all(sock_, &net_len, sizeof(net_len))) {
        perror("send length");
        return false;
      }

      if (!send_all(sock_, msg.data(), msg.size())) {
        perror("send payload");
        return false;
      }

      return true;
    }

    bool is_connected() const {
      return sock_ >= 0;
    }

    void close_conn() {
      if (sock_ >= 0) {
        close(sock_);
        sock_ = -1;
      }
    }

    ~OdinClient() {
      close_conn();
    }

  private:
    std::string path_;
    int sock_ = -1;
};

int main() {
  OdinClient client("/tmp/odin0000.socket");

  if (!client.connect_to_server()) {
    std::cerr << "Failed to connect\n";
    return 1;
  }

  std::cout << "Connected. Type messages. '/quit' to exit.\n";

  std::string input;
  while (true) {
    std::cout << "> " << std::flush;

    if (!std::getline(std::cin, input)) break;

    if (input == "/quit" || input == "exit") break;

    if (!client.send_message(input)) {
      std::cerr << "Send failed\n";
      break;
    }
  }

  client.close_conn();
  std::cout << "Disconnected.\n";
  return 0;
}
