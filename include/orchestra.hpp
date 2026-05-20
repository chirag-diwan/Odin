#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>
#include "ggufreader.hpp"
#include "model.hpp"
#include "network.hpp"


class Orchestrator{
  private:
    std::vector<uint64_t> tokens;
    std::mutex token_mutex;
    std::condition_variable token_free;
    bool is_tokens_free;

    GGufReader ggufreader;
    Model model;
    Network networkHandler;

  public:
    Orchestrator(int argc , char** argv){
       
    }

    void Start(){
      
    }
};
