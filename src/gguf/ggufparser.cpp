#include "../../include/ggufparser.hpp"
#include "../../include/gguf.hpp"


std::string_view GGufParser::parseString(){
  auto length = read_unaligned<uint64_t>(getCurrentPositionPointer());
  advanceOffset(sizeof(uint64_t));

  auto data = static_cast<uint8_t*>(getCurrentPositionPointer());
  advanceOffset(sizeof(char) * length);

  return std::string_view(reinterpret_cast<char*>(data) , length);
}

GGufArray GGufParser::parseArray() {
  GGufArray arr;

  auto element_type = static_cast<GGufValueType>(read_unaligned<uint32_t>(getCurrentPositionPointer()));
  advanceOffset(sizeof(uint32_t));

  auto element_count =
    read_unaligned<uint64_t>(getCurrentPositionPointer());
  advanceOffset(sizeof(decltype(element_count)));

  arr.length = element_count;
  arr.elemType = element_type;

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

void GGufParser::parseKeyValue() {
  auto metadata_key = parseString();
  auto value_type = read_unaligned<uint32_t>(getCurrentPositionPointer());

  advanceOffset(sizeof(decltype(value_type)));

  GGufValue parsed_value;
  parsed_value.data = static_cast<uint8_t*>(getCurrentPositionPointer());
  parsed_value.type = value_type;

  if(value_type == GGUF_VALUE_TYPE_ARRAY){
    parsed_value.array = parseArray();
  }else if(value_type == GGUF_VALUE_TYPE_STRING){
    parsed_value.string = parseString();
  }else{
    advanceOffset(GGufValueSize(value_type));
  }

  metadata_key_values_.push_back({ metadata_key , std::move(parsed_value) });
}

void GGufParser::ParseFile(int fd , void * mmap_ptr , size_t fileSize) {
  fileDescriptor_ = fd;
  mappedData_     = static_cast<uint8_t*>(mmap_ptr);
  totalSize_      = fileSize;
  byteAlignment_  = 32;
  currentOffset_  = 0;

  parseHeader();
  parseAllKeyValues();
  parseAllTensors();
}

void GGufParser::parseHeader() {
  Errorif(currentOffset_ != 0, "Offset is not zero on the first call");

  header_ = static_cast<GGufHeader*>(getCurrentPositionPointer())[0];
  advanceOffset(sizeof(decltype(header_)));
}



void GGufParser::parseAllKeyValues() {
  for (size_t i = 0; i < header_.metadataKvCount; ++i) {
    parseKeyValue();
  }

  for(const auto& kv : metadata_key_values_){
    if (kv.name == "general.alignment") {
      this->byteAlignment_ = Extract<uint64_t,GGUF_VALUE_TYPE_UINT32 ,GGUF_VALUE_TYPE_UINT64 >(
                                                                                                kv.value);
      return;
    }
  }
  this->byteAlignment_ = 32;
}

void GGufParser::parseAllTensors() {
  for (size_t i = 0; i < header_.tensorCount; ++i) {
    GGufTensor tensor;
    tensor.name = parseString();

    tensor.dimensionCount =
      read_unaligned<uint32_t>(getCurrentPositionPointer());
    advanceOffset(sizeof(uint32_t));

    for (size_t j = 0; j < tensor.dimensionCount; j++) {
      tensor.dimensions[j] =
        read_unaligned<int64_t>(getCurrentPositionPointer());
      advanceOffset(sizeof(int64_t));
    }

    tensor.tensorType= static_cast<ggml_type>(
                                               read_unaligned<uint32_t>(getCurrentPositionPointer()));
    advanceOffset(sizeof(uint32_t));

    tensor.fileOffset =
      read_unaligned<uint64_t>(getCurrentPositionPointer());
    advanceOffset(sizeof(uint64_t));

    uint64_t byte_size = 1;
    for (uint8_t k = 0; k < tensor.dimensionCount; ++k) {
      byte_size *= tensor.dimensions[k];
    }

    const auto block_size = ggml_blck_size(tensor.tensorType);
    Errorif(byte_size % block_size != 0, "Number of elements in tensor ",
            tensor.name, " is not a multiple of block size ", block_size);
    byte_size = byte_size * ggml_type_size(tensor.tensorType) / block_size;
    tensor.byteSize = byte_size;

    tensors_.push_back(tensor);
  }
  dataOffset_ = (currentOffset_ + byteAlignment_ - 1) & ~(byteAlignment_ - 1);
  for(auto& tensor : tensors_){
    tensor.fileOffset = tensor.fileOffset + dataOffset_;
    tensor.weightsData = mappedData_ + tensor.fileOffset;
  }
}
