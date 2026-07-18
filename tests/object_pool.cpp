#include "../include/data_structures/obj_pool.hpp"
#include "../include/logging.hpp"

int main(){

  object_pool<int> pool;


  std::optional<uint32_t> pos;
  for(size_t i = 0 ; i < 1024 ; i+=3){
    if(pos = pool.add(i) ; !pos.has_value()){
      Log(WARN, "Pool over flow" , i);
      break;
    }

    if(!pool.remove(*pos)){
      Log(WARN, "Index not present" , i);
      break;
    }
    if(!pool.add(i)){
      Log(WARN, "Pool over flow" , i);
      break;
    }
  }

  Log(INFO, pool.size());
}
