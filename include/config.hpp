#pragma once
#include "errors.hpp"
#include "types.hpp"
#include <cstring>
#include <string>


Config ParseConfig(int argc , char ** argv){
  Config config;
  for(int i = 0 ; i < argc ; i++){
    if(strcmp(argv[i], "--thread") == 0){
      Errorif(i + 1 > argc, "Expected thread count after --thread");
      config.thread_count = std::stoi(argv[i + 1]);
    }else if(strcmp(argv[i], "--port") == 0){
      Errorif(i + 1 > argc, "Expected port after --port");
      config.port = std::stoi(argv[i + 1]);
    }else if(strcmp(argv[i], "--model") == 0){
      Errorif(i + 1 > argc, "Expected model path after --model");
      config.model_path = argv[i + 1];
    } 
  }
  return config;
}
