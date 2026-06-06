# Odin

Odin is a CPU-optimized inference engine designed for edge hardware. The primary objective is to maximize inference throughput on resource-constrained devices using quantized models (GGUF) with minimal accuracy degradation.

## Current State

**Alpha / Active Development**
Odin currently supports the **Qwen2** architecture class. Support for additional architectures is actively being implemented.

## Performance Metrics

To generate a metric-logging build of the engine:

```bash
mkdir build && cd build
cmake .. -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DTEST_FILE=../tests/qwen_metrics.cpp
make -j$(nproc)

```

### Benchmark Results (Qwen2)

**Example Output**
```bash
================================================
             ADVANCED ENGINE METRICS
================================================
 [ EXECUTION PROFILE ]
   - Prefill Workload      : 22 tokens
   - Time-to-First-Token   : 229.058 ms
   - Prefill Speed         : 96.046 tokens/sec
   - Decode Workload       : 262 tokens
   - Total Decode Duration : 6483.314 ms
   - Decode Speed          : 40.411 tokens/sec
 -----------------------------------------------
 [ LATENCY VARIABILITY & QoS ]
   - Avg Inter-Token (ITL) : 24.745 ms/token
   - ITL Jitter (Std Dev)  : 8.646 ms
   - p50 (Median Latency)  : 19.613 ms
   - p95 (Tail Latency)    : 39.453 ms
   - p99 (Worst Spikes)    : 45.055 ms
 -----------------------------------------------
 [ HARDWARE & CONTEXT RESOURCE ]
   - Est. Memory Bandwidth : 161.645726 GB/s
   - Sequence Length       : 285 tokens
   - KV Saturation         : 0.870 %
================================================
```

**Hardware Profile:** AMD Ryzen 5 7520U (4C/8T) @ 4.38GHz

| Metric | Result |
| --- | --- |
| **Prefill Speed** | 78.29 tokens/sec |
| **Decode Speed** | 68.33 tokens/sec |
| **Median Latency (p50)** | 14.37 ms |
| **Tail Latency (p99)** | 24.36 ms |
| **Est. Memory Bandwidth** | ~273.3 GB/s |

## Installation

### Prerequisites

* A C++17 compatible compiler (GCC or Clang)
* CMake (3.14+)
* Make

### Build Instructions

```bash
git clone --recurse-submodules [https://github.com/chirag-diwan/Odin.git](https://github.com/chirag-diwan/Odin.git)
cd Odin
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

```

## Usage

Odin requires model weights in the GGUF format.

```bash
./odin --model "/path/to/qwen2-model.gguf" --thread $(nproc)

```

### CLI Arguments

| Argument | Description |
| --- | --- |
| `--model` | **Required.** Absolute or relative path to the `.gguf` model file. |
| `--thread` | **Optional.** Maximum number of threads allocated to the backend for computation. |

```
