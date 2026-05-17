test:
	cd build && \
	cmake .. && \
	make -j$(nproc) && \
	perf record ./odin && \
	perf report

report:
	cd build && \
	perf report

run:
	cd build && \
	cmake .. && \
	make -j$(nproc) && \
	./odin
