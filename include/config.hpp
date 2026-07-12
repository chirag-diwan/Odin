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
    }else if(strcmp(argv[i], "--ipc-path") == 0){
      Errorif(i + 1 > argc, "Expected path after --ipc-path");
      config.ipc_path = argv[i + 1];
    }else if(strcmp(argv[i], "--model") == 0){
      Errorif(i + 1 > argc, "Expected model path after --model");
      config.model_path = argv[i + 1];
    }else if(strcmp(argv[i], "--tokeniser-json") == 0){
      Errorif(i + 1 > argc, "Expected tokeniser json file path after --tokeniser-json");
      config.tokeniser_json_path = argv[i + 1];
    }else if(strcmp(argv[i], "--use-ipc") == 0){
      Errorif(i + 1 > argc, "Expected true or false after --use-ipc");
      if(strcmp(argv[i + 1] ,"true") == 0){
        config.use_ipc = true;
      }else{
        config.use_ipc = false;
      }
    }else if(strcmp(argv[i], "--history") == 0){
      Errorif(i + 1 > argc, "Expected file path after --history");
      config.history_path = argv[i + 1];
    }else if(strcmp(argv[i], "--use-http") == 0){
      Errorif(i + 1 > argc, "Expected true or false after --use-http");
      if(strcmp(argv[i + 1] ,"true") == 0){
        config.use_http = true;
      }else{
        config.use_http = false;
      }
    }else if(strcmp(argv[i], "--port") == 0){
      Errorif(i + 1 > argc, "Expected a valid port after --port");
      config.port = std::stoi(argv[i + 1]);
    }
  }
  return config;
}
