#include "../include/logging.hpp"
#include "../include/vector_backend_linked_list.hpp"
#include <cstdint>
#include <vector>


int main(){
  VBlist list;
  for(size_t i = 0 ; i < 100 ; i++){
    list.push(i);
  }

  for(size_t i = 0 ; i < 100 ; i+=2){
    list.erase(i);
  }

  std::vector<uint32_t> vec;
  list.push_in(vec);
  for(const auto tok : vec){
    Log(tok);
  }
}
