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
    int      file_descriptor_;

    uint64_t total_size_;

    uint64_t current_offset_;
    uint64_t data_offset_;
    uint64_t byte_alignment_;

    uint8_t* mapped_data_;



  private:
    __attribute__((always_inline)) inline void* getCurrentPositionPointer() {
      return &mapped_data_[current_offset_];
    }

    __attribute__((always_inline)) inline void advanceOffset(size_t step_size) {
      Errorif(current_offset_ + step_size > total_size_, "Size overflow");
      current_offset_ += step_size;
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
    metadatakv_t metadata_key_values_;

    GGufParser(const std::string& filepath);

    GGufParser(const GGufParser&) = delete;
    GGufParser(GGufParser&&) = default;
    GGufParser &operator=(const GGufParser&) = delete;
    GGufParser &operator=(GGufParser&&) = default;

    ~GGufParser();


    std::pair<void* , size_t> GetParsedFile() ;

};
