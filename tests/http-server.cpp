#include "../include/http/http-manager.hpp"
#include <chrono>
#include <csignal>
#include <thread>

std::sig_atomic_t interupt = false;

void sig_int_interupt(int){
  interupt = true;
}

int main(){
  HttpManager server(interupt);
  server.start_listen();
  while(!interupt){
    Log(server.read_prompt());
    for(int i = 0 ; i < 20 ; i++){
      server.write_infered("TEST");
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    //Log(server.read_prompt());
  }
  server.stop();
}
