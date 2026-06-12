#include <netinet/in.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

class stream_buffer{
  private:
    std::vector<uint8_t> buffer_;
    int read_fd_;
    size_t read_head;


  public:
    stream_buffer(int fd) :read_fd_(fd) , read_head(0){ 
     buffer_.reserve(1024);
    }

    [[nodiscard]]
      bool is_readable(size_t size){
        return read_head + size < buffer_.size();
      }


    [[nodiscard]]
      std::optional<uint8_t> read_u8(){
        if(read_head + sizeof(uint8_t)<= buffer_.size()){

          auto val = buffer_[read_head];
          read_head ++;

          return val;
        }else{
          return std::nullopt;
        }
      }


      [[nodiscard]]
      std::optional<uint32_t> read_u32(){
        if(read_head + sizeof(uint32_t) <= buffer_.size()){
          uint32_t val;

          memcpy(&val , buffer_.data() + read_head, sizeof(val));
          read_head += sizeof(val);

          val = ntohl(val);
          return val;
        }else{
          return std::nullopt;
        }
      }

    bool fill(size_t size){
      size_t total_bytes = 0;
      uint8_t buffer[256];

      while(total_bytes < size){
        memset(buffer, 0,sizeof(buffer));
        auto bytes_read = ::read(read_fd_, buffer,sizeof(buffer));
        if(bytes_read == -1 || bytes_read == 0){
          return false;
        }
        buffer_.insert(buffer_.end() , std::begin(buffer) , std::begin(buffer) + bytes_read);
        total_bytes += bytes_read;
      }
      return true;
    }


    decltype(auto) begin(){
      return buffer_.begin() + read_head;
    }

    decltype(auto) end(){
      return buffer_.end();
    }
};
