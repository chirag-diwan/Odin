#pragma once

#include "logging.hpp"
#include <atomic>
#include <cerrno>
#include <cstring>
#include <condition_variable>
#include <csignal>
#include <functional>
#include <memory>
#include <mutex>
#include <sched.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
#include <sys/fcntl.h>
#include <sys/wait.h>

class Executor{
  private:
    static constexpr size_t write_end = 1;
    static constexpr size_t read_end = 0;

    std::vector<std::thread> workers;
    std::atomic<size_t> workerCount;

    std::mutex workerCountMutex;
    std::condition_variable cv;

    std::shared_ptr<std::sig_atomic_t> interupt;

  public:
    void Init(std::shared_ptr<std::sig_atomic_t>& interupt_ptr){
      interupt = interupt_ptr;
      workerCount.store(0 , std::memory_order_release);
    }

    void ExecuteNew(const std::string& code , std::function<void(const std::string& , int)> output_callback) {
      workers.emplace_back([this, code = std::move(code) , output_callback = std::move(output_callback)] {
        int stdIn[2];
        int stdOut[2];

        if (pipe(stdIn) == -1) {
          Log(ERROR , "Unable to create stdin pipe");
          return;
        }

        if (pipe(stdOut) == -1) {
          close(stdIn[0]);
          close(stdIn[1]);
          Log(ERROR , "Unable to create stdout pipe");
          return;
        }

        workerCount.fetch_add(1, std::memory_order_relaxed);

        pid_t pid = fork();

        if (pid == 0) {
          close(stdIn[write_end]);
          close(stdOut[read_end]);

          dup2(stdIn[read_end], STDIN_FILENO);
          dup2(stdOut[write_end], STDOUT_FILENO);

          close(stdIn[read_end]);
          close(stdOut[write_end]);

          execlp( "python3", "python3", "-u", "-c", code.c_str(), nullptr);

          _exit(127);
        }

        if (pid < 0) {
          close(stdIn[read_end]);
          close(stdIn[write_end]);
          close(stdOut[read_end]);
          close(stdOut[write_end]);

          workerCount.fetch_sub(1, std::memory_order_acq_rel);
          cv.notify_all();

          Log(ERROR , "Unable to fork process\n");
          return;
        }

        close(stdIn[read_end]);
        close(stdIn[write_end]);
        close(stdOut[write_end]);

        char bytes[512];
        std::string output;

        while (true) {
          if(*interupt){
            break;
          }

          ssize_t n = read( stdOut[read_end], bytes, sizeof(bytes));

          if (n > 0) {
            output.append(bytes, n);
          } else if (n == 0) {
            break;
          } else if (errno == EINTR) {
            continue;
          } else {
            Log(ERROR , "read failed: " , strerror(errno));
            break;
          }
        }

        close(stdOut[read_end]);

        int status;
        waitpid(pid, &status, 0);
        output_callback(output , status);

        if (workerCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          cv.notify_all();
        }
      });
    }

    void Delete() {
      std::unique_lock lock(workerCountMutex);

      cv.wait(lock, [this] {
        return workerCount.load(std::memory_order_acquire) == 0;
      });

      lock.unlock();

      for (auto& worker : workers) {
        if (worker.joinable())
          worker.join();
      }
    }
};
