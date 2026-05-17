#pragma once

#include "errors.hpp"
#include "gguf.hpp"
#include "types.hpp"
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
    ModelGlobals global_struct;
    std::vector<GGufTensor> tensors;
    MetadataKV_t metadata_key_values;


  private:
    __attribute__((always_inline)) inline void* getCurrentPositionPointer() {
      return &mapped_data[current_offset];
    }

    __attribute__((always_inline)) inline void advanceOffset(size_t step_size) {
      Errorif(current_offset + step_size > total_size, "Size overflow");
      current_offset += step_size;
    }


    std::string_view parseString(){
      auto length =
        reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(uint64_t));

      auto data = static_cast<uint8_t*>(getCurrentPositionPointer());
      advanceOffset(sizeof(char) * length);

      return std::string_view(reinterpret_cast<char*>(data) , length);
    }

    GGufArray parseArray() {
      GGufArray arr;
      auto element_type =
        static_cast<GGufValueType>(reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0]);
      advanceOffset(sizeof(decltype(element_type)));

      auto element_count =
        reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(element_count)));

      arr.length = element_count;
      arr.elem_type = element_type;

      if(element_type == GGUF_VALUE_TYPE_ARRAY){
        for(size_t i = 0 ; i < element_count ; i++){
          parseArray();
        }
      }else if (element_type == GGUF_VALUE_TYPE_STRING){
        for(size_t i = 0 ; i < element_count ; i++){
          std::string_view str = parseString();
          arr.strings.emplace_back(str);
        }
      }else{
        arr.data = static_cast<uint8_t*>(getCurrentPositionPointer());
        advanceOffset((GGufValueSize(element_type)) * element_count);
      }
      return arr;
    }

    void parseKeyValue() {
      auto metadata_key = parseString();
      auto value_type =
        reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(value_type)));

      GGufValue parsed_value;
      parsed_value.data =
        reinterpret_cast<uint8_t*>(getCurrentPositionPointer());
      parsed_value.type = value_type;

      if(value_type == GGUF_VALUE_TYPE_ARRAY){
        parsed_value.array = parseArray();
      }else if(value_type == GGUF_VALUE_TYPE_STRING){
        parsed_value.string = parseString();
      }else{
        advanceOffset(GGufValueSize(value_type));
      }
      metadata_key_values.push_back({
          metadata_key ,
          parsed_value
          });
    }

  public:

    GGufReader(){
      file_descriptor = 0;
      mapped_data = nullptr;
      total_size = 0;
      current_offset = 0;
      data_offset = 0;
      byte_alignment = 0;

      header = {};
      global_struct = {};

    }

    AddrLenPair OpenFile(const char* filepath) {
      int opened_descriptor = open(filepath, O_RDONLY);
      Errorif(opened_descriptor == -1, "Not a valid file descriptor for %?",
          filepath);
      struct stat file_statistics;
      Errorif(fstat(opened_descriptor, &file_statistics) == -1,
          "Unable to get file stats for ", filepath);
      void* memory_mapped_pointer = mmap(NULL, file_statistics.st_size, PROT_READ,
          MAP_PRIVATE, opened_descriptor, 0);
      Errorif(memory_mapped_pointer == MAP_FAILED, "Mapping failed for ",
          filepath);

      this->file_descriptor = opened_descriptor;
      this->mapped_data     = reinterpret_cast<uint8_t*>(memory_mapped_pointer);
      this->total_size      = file_statistics.st_size;
      this->byte_alignment  = 32;

      return {memory_mapped_pointer , file_statistics.st_size};
    }

    void ParseHeader() {
      Errorif(current_offset != 0, "Offset is not zero on the first call");

      header =
        reinterpret_cast<struct GGufHeader*>(getCurrentPositionPointer())[0];
      advanceOffset(sizeof(decltype(header)));
    }



    void ParseAllKeyValues() {
      for (size_t i = 0; i < header.metadata_kv_count; ++i) {
        parseKeyValue();
      }

      for(const auto& kv : metadata_key_values){
        if (kv.name == "general.alignment") {
          this->byte_alignment = Extract<uint64_t,GGUF_VALUE_TYPE_UINT32 ,GGUF_VALUE_TYPE_UINT64 >(
              kv.value);
        }
      }
    }

    void ParseAllTensors() {
      for (size_t i = 0; i < header.tensor_count; ++i) {
        GGufTensor tensor;
        tensor.name = parseString();

        tensor.dimension_count =
          reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0];
        advanceOffset(sizeof(uint32_t));

        for (size_t j = 0; j < tensor.dimension_count; j++) {
          tensor.dimensions[j] =
            reinterpret_cast<int64_t*>(getCurrentPositionPointer())[0];
          advanceOffset(sizeof(int64_t));
        }

        tensor.tensor_type= static_cast<ggml_type>(
            reinterpret_cast<uint32_t*>(getCurrentPositionPointer())[0]);
        advanceOffset(sizeof(uint32_t));

        tensor.file_offset =
          reinterpret_cast<uint64_t*>(getCurrentPositionPointer())[0];
        advanceOffset(sizeof(uint64_t));

        uint32_t byte_size = 1;
        for (uint8_t i = 0; i < tensor.dimension_count; ++i) {
          byte_size *= tensor.dimensions[i];
        }

        const auto block_size = ggml_blck_size(tensor.tensor_type);
        Errorif(byte_size % block_size != 0, "Number of elements in tensor ",
            tensor.name, " is not a multiple of block size ", block_size);
        byte_size = byte_size * ggml_type_size(tensor.tensor_type) / block_size;
        tensors.push_back(tensor);
      }
      data_offset = (current_offset + byte_alignment - 1) & ~(byte_alignment - 1);
    }
    ~GGufReader(){
      munmap(mapped_data, total_size);
    }
};
