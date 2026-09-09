#pragma once

#include "./errors.hpp"
#include "./types.hpp"
#include <cstdint>
#include <fcntl.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <vector>


class GGufParser {
  private:
    int      fileDescriptor_;
    uint64_t totalSize_;
    uint64_t currentOffset_;
    uint64_t dataOffset_;
    uint64_t byteAlignment_;
    uint8_t* mappedData_;

  private:
    __attribute__((always_inline)) inline void* getCurrentPositionPointer() {
      return &mappedData_[currentOffset_];
    }

    __attribute__((always_inline)) inline void advanceOffset(size_t step_size) {
      Errorif(currentOffset_ + step_size > totalSize_, "Size overflow");
      currentOffset_ += step_size;
    }


    std::string_view parseString();

    GGufArray parseArray() ;
    void parseKeyValue() ;
    void parseHeader() ;
    void parseAllKeyValues() ;
    void parseAllTensors() ;

  public:
    GGufHeader header_;
    std::vector<GGufTensor> tensors_;
    std::vector<metadata_key_value> metadata_key_values_;

    void ParseFile(int fd , void * mmap_ptr , size_t fileSize);
};
