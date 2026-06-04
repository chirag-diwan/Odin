#include "../include/engine.hpp"
#include "../include/model_utils.hpp"
#include "../include/qwen2_tokeniser.hpp"
#include "../include/ggufreader.hpp"
#include "../include/config.hpp"
#include "../include/logging.hpp"
#include "../include/types.hpp"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <string>
#include <sys/mman.h>
#include <cmath>
#include <algorithm>
#include <iomanip>

using clock_type = std::chrono::steady_clock;

struct AdvancedMetrics {
  clock_type::time_point prefill_start;
  clock_type::time_point prefill_end;

  std::vector<clock_type::time_point> decode_timestamps;
  size_t prefill_token_count = 0;
  size_t model_bytes_per_token = 0; // Set this based on model size (Weights + KV overhead)

  void Reset(size_t bytes_per_token = 0) {
    decode_timestamps.clear();
    prefill_token_count = 0;
    model_bytes_per_token = bytes_per_token;
  }

  void StartPrefill(size_t input_tokens) {
    prefill_token_count = input_tokens;
    prefill_start = clock_type::now();
  }

  void EndPrefill() {
    prefill_end = clock_type::now();
    decode_timestamps.push_back(prefill_end); 
  }

  void RecordDecodeToken() {
    decode_timestamps.push_back(clock_type::now());
  }

  void Report(size_t total_kv_slots) const {
    if (decode_timestamps.size() < 2) return;

    using std::chrono::duration_cast;
    using std::chrono::microseconds;

    double prefill_ms = duration_cast<microseconds>(prefill_end - prefill_start).count() / 1000.0;
    double total_decode_ms = duration_cast<microseconds>(decode_timestamps.back() - decode_timestamps.front()).count() / 1000.0;
    size_t decode_tokens_count = decode_timestamps.size() - 1;

    std::vector<double> itl_ms;
    itl_ms.reserve(decode_tokens_count);
    for (size_t i = 1; i < decode_timestamps.size(); ++i) {
      itl_ms.push_back(duration_cast<microseconds>(decode_timestamps[i] - decode_timestamps[i-1]).count() / 1000.0);
    }

    double sum_itl = std::accumulate(itl_ms.begin(), itl_ms.end(), 0.0);
    double avg_itl = sum_itl / itl_ms.size();

    double variance_sum = 0.0;
    for (double t : itl_ms) {
      variance_sum += (t - avg_itl) * (t - avg_itl);
    }
    double std_dev_itl = std::sqrt(variance_sum / itl_ms.size());

    std::vector<double> sorted_itl = itl_ms;
    std::sort(sorted_itl.begin(), sorted_itl.end());
    double p50 = sorted_itl[static_cast<size_t>(sorted_itl.size() * 0.50)];
    double p95 = sorted_itl[static_cast<size_t>(sorted_itl.size() * 0.95)];
    double p99 = sorted_itl[static_cast<size_t>(sorted_itl.size() * 0.99)];

    double prefill_tps = prefill_ms > 0 ? (prefill_token_count / (prefill_ms / 1000.0)) : 0.0;
    double decode_tps = total_decode_ms > 0 ? (decode_tokens_count / (total_decode_ms / 1000.0)) : 0.0;

    double achieved_gbps = 0.0;
    if (model_bytes_per_token > 0 && total_decode_ms > 0) {
      double total_bytes_moved = static_cast<double>(model_bytes_per_token) * decode_tokens_count;
      achieved_gbps = (total_bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (total_decode_ms / 1000.0);
    }

    size_t final_context = prefill_token_count + decode_tokens_count + 1;
    double kv_saturation = total_kv_slots > 0 ? (static_cast<double>(final_context) / total_kv_slots) * 100.0 : 0.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n================================================\n";
    std::cout << "             ADVANCED ENGINE METRICS            \n";
    std::cout << "================================================\n";
    std::cout << " [ EXECUTION PROFILE ]\n";
    std::cout << "   - Prefill Workload      : " << prefill_token_count << " tokens\n";
    std::cout << "   - Time-to-First-Token   : " << prefill_ms << " ms\n";
    std::cout << "   - Prefill Speed         : " << prefill_tps << " tokens/sec\n";
    std::cout << "   - Decode Workload       : " << decode_tokens_count << " tokens\n";
    std::cout << "   - Total Decode Duration : " << total_decode_ms << " ms\n";
    std::cout << "   - Decode Speed          : " << decode_tps << " tokens/sec\n";
    std::cout << " -----------------------------------------------\n";
    std::cout << " [ LATENCY VARIABILITY & QoS ]\n";
    std::cout << "   - Avg Inter-Token (ITL) : " << avg_itl << " ms/token\n";
    std::cout << "   - ITL Jitter (Std Dev)  : " << std_dev_itl << " ms\n";
    std::cout << "   - p50 (Median Latency)  : " << p50 << " ms\n";
    std::cout << "   - p95 (Tail Latency)    : " << p95 << " ms\n";
    std::cout << "   - p99 (Worst Spikes)    : " << p99 << " ms\n";
    std::cout << " -----------------------------------------------\n";
    std::cout << " [ HARDWARE & CONTEXT RESOURCE ]\n";
    std::cout << "   - Est. Memory Bandwidth : " << (model_bytes_per_token > 0 ? std::to_string(achieved_gbps) + " GB/s" : "N/A") << "\n";
    std::cout << "   - Sequence Length       : " << final_context << " tokens\n";
    std::cout << "   - KV Saturation         : " << kv_saturation << " %\n";
    std::cout << "================================================\n\n";
  }
};


int main(int argc , char **argv) {
  if(argc < 2){
    Log("Usage : \n\t ./odin --model /path/to/model --thread 3 \n");
    return 0;
  }
  Config config = ParseConfig(argc, argv);

  GGufReader reader;

  auto [addr , len]  = reader.OpenFile(config.model_path);

  reader.ParseHeader();
  reader.ParseAllKeyValues();
  reader.ParseAllTensors();


  ggml_backend_t backend = ggml_backend_cpu_init();
  ggml_backend_cpu_set_n_threads(backend, config.thread_count);

  ggml_init_params static_ctx_params = {
    .mem_size = 10 * 1024 * 1024,
    .mem_buffer = NULL,
    .no_alloc = true 
  };

  ggml_context* static_ctx = ggml_init(static_ctx_params);


  auto globals = GetModelGlobals(reader.metadata_key_values);
  auto model_ = CreateModel(static_ctx, reader);

  Engine engine(model_ , static_ctx , backend);
  engine.ReservePrefillMemory();
  engine.ReserveDecodeMemory();


  QwenStyleTokenizer tokeniser(globals);

  std::vector<uint32_t> tokens;

  size_t last_index = 0;
  bool infer_complete = true;

  AdvancedMetrics metrics;
  size_t max_kv_slots = globals.context_length;

  size_t model_bytes = 4294967296; 


  while(true){
    if (infer_complete) {
      std::string prompt;
      std::cout << "\n $ ";
      std::getline(std::cin, prompt);

      if (prompt == "exit") {
        break;
      }

      metrics.Reset(model_bytes);
      last_index = tokens.size();

      tokeniser.TokeniseFormatted(prompt, tokens);
      span<uint32_t> tokens_view(tokens.data() + last_index, tokens.size() - last_index);

      metrics.StartPrefill(tokens_view.size());
      auto next_token = engine.Prefill(tokens_view);
      metrics.EndPrefill();

      tokens.emplace_back(next_token);
      tokeniser.Decode(next_token);

      infer_complete = false;
    } else {  
      auto next_token = engine.Infer(tokens.back());

      metrics.RecordDecodeToken();
      tokens.emplace_back(next_token);

      if(next_token == globals.ggml_eos_token_id){
        infer_complete = true;
        metrics.Report(max_kv_slots);
        continue;
      }

      tokeniser.Decode(next_token);
    }
  }

  ggml_free(static_ctx);
  munmap(addr, len);
  return 0;
}
