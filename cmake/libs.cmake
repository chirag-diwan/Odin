add_library(gguf
  ${CMAKE_SOURCE_DIR}/src/gguf/gguf.cpp
  ${CMAKE_SOURCE_DIR}/src/gguf/ggufreader.cpp
)

add_library(http
  ${CMAKE_SOURCE_DIR}/src/http/http-manager.cpp
)  

add_library(ipc
  ${CMAKE_SOURCE_DIR}/src/ipc/ipc_manager.cpp
)

add_library(tokeniser
  ${CMAKE_SOURCE_DIR}/src/tokeniser/json_tokeniser.cpp
)

add_library(engine
  ${CMAKE_SOURCE_DIR}/src/engine/engine_impl.cpp
  ${CMAKE_SOURCE_DIR}/src/engine/forward.cpp
)

add_library(common
  ${CMAKE_SOURCE_DIR}/src/block.cpp
  ${CMAKE_SOURCE_DIR}/src/welcome.cpp
  ${CMAKE_SOURCE_DIR}/src/stream_buffer.cpp
)

add_library(simdjson
  ${CMAKE_SOURCE_DIR}/external/simdjson/simdjson.cpp
)

target_link_libraries(engine PUBLIC gguf tokeniser)
