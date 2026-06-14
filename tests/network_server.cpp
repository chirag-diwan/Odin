#include <ios>
#include <iostream>
#include "../include/network/network_manager.hpp"
#include "../include/logging.hpp"

int main() {
  NetworkManager manager;

  std::cout << std::unitbuf;
  manager.start_listen();

  while (true) {
    auto prompt = manager.read_prompt();

    if (!prompt.has_value()) {
      continue;
    }

    if (*prompt == "!exit") {
      break;
    }
    Log(*prompt);
  }

  return 0;
}
