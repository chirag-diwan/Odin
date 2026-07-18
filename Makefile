.PHONY: debug release run-test run run-server

TEST ?=
ENABLETEST ?= OFF
model ?= 
tokeniser ?=

debug:
	@cmake -S . -B build/debug \
		-DCMAKE_BUILD_TYPE=Debug \
		-DENABLE_TESTS=$(ENABLETEST) \
		-DTEST_FILE=$(TEST)
	@cmake --build build/debug

release:
	@cmake -S . -B build/release \
		-DCMAKE_BUILD_TYPE=Release \
		-DENABLE_TESTS=$(ENABLETEST) \
		-DTEST_FILE=$(TEST)
	@cmake --build build/release

run:
	cd build/release && ./odin --model $(model) --tokeniser-json $(tokeniser)
	
run-server:
	cp -r interface build
	cd build/release && ./odin-http-server --model $(model) --tokeniser-json $(tokeniser)

run-test:
ifeq ($(ENABLETEST),ON)
	@if [ -z "$(TEST)" ]; then \
		echo "Error: TEST is not set."; \
		exit 1; \
	fi
	@cd $(BUILD_DIR) && ./$(basename $(notdir $(TEST)))_TEST
else
	@echo "Tests are disabled (ENABLETEST=$(ENABLETEST))."
endif
