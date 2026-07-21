#include "../include/http-manager.hpp"
#include "../include/logging.hpp"
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
  PromptReq prompt;
  while(!interupt){
    prompt = server.read_prompt();
    Log("Content" , prompt.content);
    for(int i = 0 ; i < 5 ; i++){
      server.write_infered("TEST");
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    server.write_infered(HttpManager::DONE_TOK);
  }
  server.stop();
}
