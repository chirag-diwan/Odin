#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
class Chanel{
  private:
    std::queue<uint64_t> data;
    std::mutex data_mutex;
    std::condition_variable cv;
  public:
    Chanel(){
    }

    void push(uint64_t token){
      std::unique_lock<std::mutex> lck(data_mutex);
      data.push(token);
      cv.notify_one();
    }

    uint64_t pop(){
      std::unique_lock<std::mutex> lck(data_mutex);
      cv.wait(lck , [this](){
        return data.size() > 0;
      });
      auto temp =  data.back();
      data.pop();
      return temp;
    }
};
