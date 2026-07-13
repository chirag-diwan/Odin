#pragma once

#include "./errors.hpp"
#include "./types.hpp"
#include <cstdint>
#include <fcntl.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>


class GGufReader {
  private:
    int      file_descriptor;
    uint8_t* mapped_data;

    uint64_t total_size;

    uint64_t current_offset;
    uint64_t data_offset;
    uint64_t byte_alignment;



  public:
    GGufHeader header;
    std::vector<GGufTensor> tensors;
    metadatakv_t metadata_key_values;


  private:
    __attribute__((always_inline)) inline void* getCurrentPositionPointer() {
      return &mapped_data[current_offset];
    }

    __attribute__((always_inline)) inline void advanceOffset(size_t step_size) {
      Errorif(current_offset + step_size > total_size, "Size overflow");
      current_offset += step_size;
    }


    std::string_view parseString();

    GGufArray parseArray() ;

    void parseKeyValue() ;

  public:
    GGufReader();

    std::pair<void* , size_t> OpenFile(const std::string& filepath) ;

    void ParseHeader() ;

    void ParseAllKeyValues() ;

    void ParseAllTensors() ;
    ~GGufReader();
};
