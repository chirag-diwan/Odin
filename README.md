# Odin
Odin is a CPU focused inference engine . The main goal of this project is the increase the inference speed on any edge hardware without compromising the model paramaters much

# Current State
Odin is in its initial development stage and only supports qwen2 class of architecture . More architecture support is being actively developed

# Metrics
SEE `bench_log.txt` for more the bench 
``` bash
[INFO]TIME PER TASK
Parsing the gguf file 11 ms
Initializing the model 1 ms
Setting up the tokeniser 106 ms
Time to first token 368 ms
Tokens per second 10 tokens/s
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
