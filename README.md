# Odin

![Odin Showcase](./assets/showcase.png)


![Odin Browser Interface Showcase](./assets/showcase_brow.png)

Odin is a CPU-optimized inference engine designed for hosting on edge hardware. The primary objective is to maximize inference throughput on resource-constrained devices using quantized models (GGUF) with minimal accuracy degradation.

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
git clone --recurse-submodules https://github.com/chirag-diwan/Odin.git
cd Odin
mkdir build && make release

```

## Usage

Odin requires model weights in the GGUF format.

```bash
./odin --model "/path/to/model.gguf" --thread $(nproc) --tokeniser-json "/path/to/tokeniser/json" --use-ipc false --ipc-path /tmp/odin0000.socket
```

## HTTP server
The HTTP server is hosted at `localhost:{port}` . The API interface for the engine is **OPENAI Compatible**. **The server has SINGLE client support**.

The API is OpenAI compatible see `docs/api.md` for more information over the API.

```bash
./odin-http-server --model "/path/to/model.gguf" --tokeniser-json "/path/to/tokeniser/json" --port 8080
```

### CLI Arguments

| Argument | Description |
| --- | --- |
| `--model` | **Required.** Absolute or relative path to the `.gguf` model file. |
| `--tokeniser-json` | **Required.** The path to tokeniser.json for the specific model you are using. |
| `--thread` | **Optional.** Maximum number of threads allocated to the backend for computation. |
| `--ipc-path` | **Optional** Path or endpoint for ipc input/output.|
| `--use-ipc` |**Optional** Enable/disable ipc mode (false = off, true = on).|
| `--port` |**(For odin-http-server) Optional** Set the port for hosting the http server(default `8080`).|


## Current State

**Alpha / Active Development**
Odin currently supports the **Qwen2** and **LLama3** architecture class. Support for additional architectures is actively being implemented.
