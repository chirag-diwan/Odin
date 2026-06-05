# Odin
Odin is a CPU focused inference engine . The main goal of this project is the increase the inference speed on any edge hardware without compromising the model paramaters much

# Current State
Odin is in its initial development stage and only supports qwen2 class of architecture . More architecture support is being actively developed

# Metrics
```bash
  cmake .. -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DTEST_FILE=../tests/qwen_metrics.cpp && make -j$(nproc)
```
The above command is used to create the metric logging version of the engine . 
The output for *Release* build on my system is given below.

``` bash
================================================
             ADVANCED ENGINE METRICS
================================================
 [ EXECUTION PROFILE ]
   - Prefill Workload      : 29 tokens
   - Time-to-First-Token   : 370.412 ms
   - Prefill Speed         : 78.291 tokens/sec
   - Decode Workload       : 461 tokens
   - Total Decode Duration : 6746.570 ms
   - Decode Speed          : 68.331 tokens/sec
 -----------------------------------------------
 [ LATENCY VARIABILITY & QoS ]
   - Avg Inter-Token (ITL) : 14.634 ms/token
   - ITL Jitter (Std Dev)  : 1.447 ms
   - p50 (Median Latency)  : 14.371 ms
   - p95 (Tail Latency)    : 16.174 ms
   - p99 (Worst Spikes)    : 24.369 ms
 -----------------------------------------------
 [ HARDWARE & CONTEXT RESOURCE ]
   - Est. Memory Bandwidth : 273.324074 GB/s
   - Sequence Length       : 491 tokens
   - KV Saturation         : 1.498 %
================================================
```

Ran on CPU with the following specifications.

```bash
Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             44 bits physical, 48 bits virtual
  Byte Order:                Little Endian
CPU(s):                      8
  On-line CPU(s) list:       0-7
Vendor ID:                   AuthenticAMD
  Model name:                AMD Ryzen 5 7520U with Radeon Graphics
    CPU family:              23
    Model:                   160
    Thread(s) per core:      2
    Core(s) per socket:      4
    Socket(s):               1
    Stepping:                0
    Microcode version:       0x8a00009
    Frequency boost:         enabled
    CPU(s) scaling MHz:      56%
    CPU max MHz:             4386.4722
    CPU min MHz:             422.7930
    BogoMIPS:                5589.11
```

# Installation
``` bash
    git clone https://github.com/chirag-diwan/Odin.git
    cd Odin
    mkdir build
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

# Running

``` bash
    ./odin --model "/path/to/model.gguf" --thread $(nproc) 
```


Cli flag and there meaning

| Flag | What for |
| --- | --- |
| --model | To specify the model path |
| --thread | The mazimum number of thread that the backend should use |
