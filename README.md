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
