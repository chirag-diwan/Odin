#include "../include/ipc/ipc_manager.hpp"

int main() {
  IPCManager server("/tmp/odin0000.socket");

  server.start_listen();

  while (true) {
    Log(server.read_prompt());
    server.write_infered("THIS IS JUST A TEST");
    server.write_infered("THIS IS JUST A TEST 2");
    server.write_infered("THIS IS JUST A TEST 3");
  }

  return 0;
}
