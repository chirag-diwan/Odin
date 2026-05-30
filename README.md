# Odin
Odin is a CPU focused inference engine . The main goal of this project is the increase the inference speed on any edge hardware without compromising the model paramaters much

# Current State
Odin is in its initial development stage and only supports qwen2 class of architecture . More architecture support is being actively developed

# Installation
``` bash
    git clone https://github.com/chirag-diwan/Odin.git
    cd Odin
    mkdir build
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

# Running

``` bash
    ./odin --model "/path/to/model.gguf" --thread $(nproc) --interactive true --prompt "Hello how are :wyou"
```

Cli flag and there meaning

| Flag | What for |
| --- | --- |
| --model | To specify the model path |
| --thread | The mazimum number of thread that the backend should use |
| --interactive | If on then the engine runs like a chat client |
| --prompt | If `--interactive false` then , used to provide prompt for non interactive chat |
