# Odin
Odin is a CPU focused inference engine . The main goal of this project is the increase the inference speed on any edge hardware without compromising the model paramaters much

# Current State
Odin is in its initial development stage and only supports qwen2 class of architecture . More architecture support is being actively developed

# Metrics

    ```bash
cmake .. -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DTEST_FILE=../tests/qwen_metrics.cpp && make -j$(nproc)
    ```
    The binary produced using the above command was used for testing . Below is the output showcasing the result.

    ``` bash
    ================================================
    ADVANCED ENGINE METRICS
    ================================================
    [ EXECUTION PROFILE ]
    - Prefill Workload      : 19 tokens
    - Time-to-First-Token   : 190.394 ms
    - Prefill Speed         : 99.793 tokens/sec
    - Decode Workload       : 27 tokens
    - Total Decode Duration : 442.744 ms
    - Decode Speed          : 60.983 tokens/sec
    -----------------------------------------------
    [ LATENCY VARIABILITY & QoS ]
    - Avg Inter-Token (ITL) : 16.397 ms/token
                              - ITL Jitter (Std Dev)  : 0.095 ms
                                                        - p50 (Median Latency)  : 16.389 ms
                                                                                  - p95 (Tail Latency)    : 16.561 ms
                                                                                                            - p99 (Worst Spikes)    : 16.663 ms
                                                                                                                                      -----------------------------------------------
                                                                                                                                      [ HARDWARE & CONTEXT RESOURCE ]
                                                                                                                                      - Est. Memory Bandwidth : 243.933289 GB/s
                                                                                                                                      - Sequence Length       : 47 tokens
                                                                                                                                      - KV Saturation         : 0.143 %
                                                                                                                                      ================================================
                                                                                                                                      ```

                                                                                                                                      The above metrics are based on my CPU with the following specifications

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
