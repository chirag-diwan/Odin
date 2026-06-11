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
    }else if(strcmp(argv[i], "--network-path") == 0){
      Errorif(i + 1 > argc, "Expected port after --network-path");
      config.network_path = argv[i + 1];
    }else if(strcmp(argv[i], "--model") == 0){
      Errorif(i + 1 > argc, "Expected model path after --model");
      config.model_path = argv[i + 1];
    }else if(strcmp(argv[i], "--temp") == 0){
      Errorif(i + 1 > argc, "Expected float temp after --temp");
      config.temperature = std::stof(argv[i + 1]);
    }else if(strcmp(argv[i], "--top-k") == 0){
      Errorif(i + 1 > argc, "Expected positive integer after --top-k");
      config.k = std::stoi(argv[i + 1]);
    }else if(strcmp(argv[i], "--tokeniser-json") == 0){
      Errorif(i + 1 > argc, "Expected tokeniser json file path after --tokeniser-json");
      config.tokeniser_json_path = argv[i + 1];
    }else if(strcmp(argv[i], "--use-network") == 0){
      Errorif(i + 1 > argc, "Expected boolean after --use-network");
      config.use_network = argv[i + 1];
    }
  }
  return config;
}
