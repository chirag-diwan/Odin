.PHONY: debug release run-test run-debug run-server-debug run run-server perf-engine perf-server run-ipc run-ipc-debug

test ?=
model ?= ~/Models/Llama-3.2-1B-Instruct-Q4_0.gguf
tokeniser ?= ~/Models/llama3tok.json
port ?= 8080

ENABLETEST ?= OFF

ifneq ($(strip $(test)),)
ENABLETEST := ON
endif

debug:
	@cmake -S . -B build/debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DENABLE_TESTS=$(ENABLETEST) \
		-DTEST_FILE=$(test)
	@cmake --build build/debug -j

release:
	@cmake -S . -B build/release \
		-DCMAKE_BUILD_TYPE=Release \
		-DENABLE_TESTS=$(ENABLETEST) \
		-DTEST_FILE=$(test)
	@cmake --build build/release -j



run-debug:
	./build/debug/odin --model $(model) --tokeniser-json $(tokeniser)
	
run-server-debug:
	./build/debug/odin-http-server --model $(model) --tokeniser-json $(tokeniser) --port $(port)

run-ipc-debug:
	./build/debug/odin-ipc-server --model $(model) --tokeniser-json $(tokeniser) --port $(port)

run:
	./build/release/odin --model $(model) --tokeniser-json $(tokeniser)
	
run-server:
	./build/release/odin-http-server --model $(model) --tokeniser-json $(tokeniser) --port $(port)

run-ipc:
	./build/release/odin-ipc-server --model $(model) --tokeniser-json $(tokeniser)
	
run-test:
ifeq ($(ENABLETEST),ON)
	@if [ -z "$(test)" ]; then \
		echo "Error: test is not set."; \
		exit 1; \
	fi
	@cd $(BUILD_DIR) && ./$(basename $(notdir $(test)))_TEST
else
	@echo "Tests are disabled (ENABLETEST=$(ENABLETEST))."
endif

perf-engine:
	perf record ./build/debug/odin --model $(model) --tokeniser-json $(tokeniser)

perf-server:
	perf record ./build/debug/odin-http-server --model $(model) --tokeniser-json $(tokeniser) --port $(port)
