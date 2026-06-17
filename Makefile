MODEL := /home/chirag/Models/Llama-3.2-1B-Instruct-Q4_0.gguf
TOKENISER := /home/chirag/Models/llama3tok.json

BUILD_DIR := build
THREADS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)
TYPE = Debug

CMAKE := cmake .. -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=$(TYPE)

.PHONY: run server 

run:
	./build/odin --model $(MODEL) --tokeniser-json $(TOKENISER) --thread $(THREADS) --use-network false

server:
	./build/network_server_TEST


.PHONY: build-server build

build-server:
	cd $(BUILD_DIR) && $(CMAKE) -DTEST_FILE=../tests/network_server.cpp && $(MAKE) -j$(THREADS)


build:
	cd $(BUILD_DIR) && $(CMAKE) -DENABLE_TESTS=OFF && $(MAKE) -j$(THREADS)

