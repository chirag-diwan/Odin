#include "../include/network/multiclient/network_manager.hpp"
#include "../include/logging.hpp"

int main() {
  NetworkManager manager;

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

    manager.write_infered("THIS IS A TEST TOKEN");
  }

  return 0;
}
