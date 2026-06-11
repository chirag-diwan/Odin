#include "../include/network/network_manager.hpp"
#include "logging.hpp"

int main(){
  NetworkManager manager;
  manager.start_listen();
  while(true){
    auto prompt = manager.read_input();
    if(prompt.has_value()){
      if(*prompt == "!exit"){
        break;
      }else{
        Log(*prompt);
      }
    }
  }
}
