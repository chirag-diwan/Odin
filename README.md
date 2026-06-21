# Odin

Odin is a CPU-optimized inference engine designed for edge hardware. The primary objective is to maximize inference throughput on resource-constrained devices using quantized models (GGUF) with minimal accuracy degradation.

**Hardware Profile:** AMD Ryzen 5 7520U (4C/8T) @ 4.38GHz , Qwen 2.5 Instruct (0.5 billion parameter)

| Metric | Result |
| --- | --- |
| **Prefill Speed** | 78.29 tokens/sec |
| **Decode Speed** | 68.33 tokens/sec |
| **Median Latency (p50)** | 14.37 ms |
| **Tail Latency (p99)** | 24.36 ms |
| **Est. Memory Bandwidth** | ~30.3 GB/s |

## Installation

### Prerequisites

* A C++20 compatible compiler (GCC or Clang)
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
./odin --model "/path/to/model.gguf" --thread $(nproc) --tokeniser-json "/path/to/tokeniser/json" --use-network false --network-path /tmp/odin0000.socket
```

### CLI Arguments

| Argument | Description |
| --- | --- |
| `--model` | **Required.** Absolute or relative path to the `.gguf` model file. |
| `--tokeniser-json` | **Required.** The path to tokeniser.json for the specific model you are using. |
| `--thread` | **Optional.** Maximum number of threads allocated to the backend for computation. |
|`--network-path` | **Optional** Path or endpoint for network input/output.|
|`--use-network` |**Optional** Enable/disable network mode (false = off, true = on).|



## Current State

**Alpha / Active Development**
Odin currently supports the **Qwen2** and **LLama3** architecture class. Support for additional architectures is actively being implemented.
Support for more architecture is being developed currently
