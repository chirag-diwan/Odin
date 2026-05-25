#include "../include/contagious_linked_list.hpp"
#include <iostream>

int main(){
  c_list<1024> list;
  for(int i = 0 ; i < 10000 ; i++){
    if(!list.push(i))break;
  }

  for(int i = 0 ; i < 10000 ; i++){
    list.erase(i);
  }

  for(const auto e : list){
    std::cout << e.val_ << '\n';
  }
}
