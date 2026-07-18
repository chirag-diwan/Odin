#include "../include/multiclient-httpmanager.hpp"
#include "../include/logging.hpp"
#include <csignal>

std::sig_atomic_t interupt;

void sig_int_interupt(int){
  interupt = true;
}

int main(){
  std::signal(SIGINT , sig_int_interupt);

  MultiClientServer server(interupt , 8080);
  server.start_listen();
  
  while(!interupt){
    Log(server.read_request());
  }
  server.stop();
}
