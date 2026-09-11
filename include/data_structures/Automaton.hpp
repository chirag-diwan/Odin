#include "definations.hpp"
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <queue>
#include <string>
#include <vector>

class Automaton{
  private:
    static constexpr size_t range = 128;

    struct Node{
      uint8_t val;
      int16_t suffixLink;

      std::array<int16_t ,range> child;
      std::vector<size_t> output;

      Node(uint8_t val) : val(val) {
        std::fill(child.begin(), child.end(),-1);
      }
    };


    static constexpr auto initRange = []()-> std::array<int16_t, range>{
      std::array<int16_t, range> a{};
      a.fill(-1);
      return a;
    }();

    int16_t root = -1;

    std::vector<std::string> patterns;
    std::vector<Node> nodes;

    ODIN_INLINE int16_t& child(int16_t parentIdx , int16_t idx){
      return nodes[parentIdx].child[idx];
    }

    ODIN_INLINE Node& childAt(int16_t parentIdx , int16_t idx){
      return nodes[child(parentIdx , idx)];
    }

    ODIN_INLINE size_t createNode(uint8_t val){
      auto temp = nodes.size();
      nodes.emplace_back(val);
      return temp;
    }

  public:

    void Insert(const std::string_view& pattern){
      size_t index = 0;

      if(root == -1){
        root = createNode(pattern[index]);
        index++;
      }

      auto current = root;

      while(index < pattern.size()){
        auto childIdx = child(current , pattern[index]);

        if(childIdx == -1){

          auto parent = current;
          auto newNode = createNode(pattern[index]);

          child(parent , pattern[index]) = newNode;
          current = newNode;

          index++;
        }else if(nodes[childIdx].val == pattern[index]){
          current = childIdx;
          index++;
        }
      }

      auto patternIdx = patterns.size();
      patterns.push_back(std::string{pattern.data() , pattern.size()});
      if(std::find(nodes[current].output.begin() , nodes[current].output.end() , patternIdx) == nodes[current].output.end()){
        nodes[current].output.push_back(patternIdx);
      }
    }

    void Insert(const std::vector<std::string>& p){
      for(const auto& pattern : p){
        size_t index = 0;

        if(root == -1){
          root = createNode(pattern[index]);
          index++;
        }

        auto current = root;

        while(index < pattern.size()){
          auto childIdx = child(current , pattern[index]);

          if(childIdx == -1){

            auto parent = current;
            auto newNode = createNode(pattern[index]);

            child(parent , pattern[index]) = newNode;
            current = newNode;

            index++;
          }else if(nodes[childIdx].val == pattern[index]){
            current = childIdx;
            index++;
          }
        }

        auto patternIdx = patterns.size();
        patterns.push_back(pattern);
        if(std::find(nodes[current].output.begin() , nodes[current].output.end() , patternIdx) == nodes[current].output.end()){
          nodes[current].output.push_back(patternIdx);
        }
      }
    }


    void BuildSufixLinks(){
      nodes[root].suffixLink = root;

      std::queue<int16_t> unvisited;
      for(size_t i = 0 ; i < range ; i++){
        if(child(root , i) != -1){
          childAt(root , i).suffixLink = root;
          unvisited.push(child(root , i));
        }
      }

      while (!unvisited.empty()){
        auto current = unvisited.front();unvisited.pop(); 
        for(size_t i = 0 ; i < range ; i++){
          auto childIdx = child(current , i);
          if(childIdx == -1)continue;

          unvisited.push(childIdx);

          auto ch = nodes[childIdx].val;

          auto link = nodes[current].suffixLink;
          int16_t foundchild = -1;

          while(true){
            foundchild = child(link , i);
            if((foundchild != -1 && nodes[foundchild].val == ch) || link == root) break;
            link = nodes[link].suffixLink; 
          }

          auto suffixLink = foundchild == -1 ? root : foundchild;
          nodes[childIdx].suffixLink = suffixLink;

          for(const auto& o : nodes[suffixLink].output){


            if(std::find(nodes[current].output.begin() , nodes[current].output.end() , o) == nodes[current].output.end()){
              nodes[current].output.push_back(o);
            }
          }
        }
      }
    }



    void Search(const std::string& str , std::vector<std::pair<size_t , std::string_view>>& out){
      auto current = root;
      size_t idx = 0;

      while(idx < str.size()){
        uint8_t ch = str[idx]; 
        auto childIdx = child(current , ch);

        if(childIdx != -1){
          current = childIdx;
          idx++;
          if(nodes[current].output.size() > 0){
            for(const auto& o : nodes[current].output){
              const auto& str = patterns[o];
              out.emplace_back(idx - str.size() , str);
            }
          }
        }else if(current == root){
          idx++;
        }else{
          current = nodes[current].suffixLink;
        }
      }
    }  

    void ToBinary(const std::string& path) {
      std::ofstream out{path, std::ios::binary};

      const uint32_t version = 1;
      const uint32_t nodeCount = static_cast<uint32_t>(nodes.size());
      const uint32_t patternCount = static_cast<uint32_t>(patterns.size());

      out.write(reinterpret_cast<const char*>(&version), sizeof(version));

      int16_t rootValue = root;
      out.write(reinterpret_cast<const char*>(&rootValue), sizeof(rootValue));

      out.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
      out.write(reinterpret_cast<const char*>(&patternCount), sizeof(patternCount));

      for (const auto& n : nodes) {
        out.write(reinterpret_cast<const char*>(&n.val), sizeof(n.val));
        out.write(reinterpret_cast<const char*>(&n.suffixLink),
                  sizeof(n.suffixLink));

        for (const auto c : n.child) {
          out.write(reinterpret_cast<const char*>(&c), sizeof(c));
        }

        uint32_t outputCount =
          static_cast<uint32_t>(n.output.size());

        out.write(reinterpret_cast<const char*>(&outputCount),
                  sizeof(outputCount));

        for (size_t o : n.output) {
          uint32_t index = static_cast<uint32_t>(o);

          out.write(reinterpret_cast<const char*>(&index),
                    sizeof(index));
        }
      }

      for (const auto& pattern : patterns) {
        uint64_t length = pattern.size();

        out.write(reinterpret_cast<const char*>(&length),
                  sizeof(length));

        out.write(pattern.data(), static_cast<std::streamsize>(length));
      }
      out.close();
    }


    void Load(const std::string& path) {
      std::ifstream in(path, std::ios::binary);

      nodes.clear();
      patterns.clear();
      root = -1;

      uint32_t version;
      in.read(reinterpret_cast<char*>(&version), sizeof(version));

      if (version != 1) {
        return;
      }

      int16_t rootValue;
      in.read(reinterpret_cast<char*>(&rootValue), sizeof(rootValue));
      root = rootValue;

      uint32_t nodeCount;
      uint32_t patternCount;

      in.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
      in.read(reinterpret_cast<char*>(&patternCount), sizeof(patternCount));

      nodes.reserve(nodeCount);
      patterns.reserve(patternCount);

      for (uint32_t i = 0; i < nodeCount; ++i) {
        Node n{0};

        in.read(reinterpret_cast<char*>(&n.val), sizeof(n.val));
        in.read(reinterpret_cast<char*>(&n.suffixLink),
                sizeof(n.suffixLink));

        for (auto& c : n.child) {
          in.read(reinterpret_cast<char*>(&c), sizeof(c));

          if (!in) {
            return;
          }
        }

        uint32_t outputCount;
        in.read(reinterpret_cast<char*>(&outputCount),
                sizeof(outputCount));

        n.output.resize(outputCount);

        for (auto& o : n.output) {
          uint32_t index;
          in.read(reinterpret_cast<char*>(&index), sizeof(index));
          o = index;
        }

        nodes.emplace_back(std::move(n));
      }

      for (uint32_t i = 0; i < patternCount; ++i) {
        uint64_t length;

        in.read(reinterpret_cast<char*>(&length), sizeof(length));

        std::string pattern(length, '\0');

        in.read(pattern.data(),
                static_cast<std::streamsize>(length));

        patterns.emplace_back(std::move(pattern));
      }

      in.close();
    }
};

