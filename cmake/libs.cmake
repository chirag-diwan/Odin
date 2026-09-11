add_library(odin_pch INTERFACE)

target_precompile_headers(odin_pch INTERFACE
  ${CMAKE_SOURCE_DIR}/external/httplib/httplib.h
  ${CMAKE_SOURCE_DIR}/external/nlohmann/json.hpp
  ${CMAKE_SOURCE_DIR}/external/jonathan/cpptui.hpp
)

target_include_directories(odin_pch INTERFACE
  ${CMAKE_SOURCE_DIR}/external
)

add_library(gguf
  ${CMAKE_SOURCE_DIR}/src/gguf/gguf.cpp
  ${CMAKE_SOURCE_DIR}/src/gguf/ggufparser.cpp
)

target_include_directories(gguf PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external
)

add_library(http
  ${CMAKE_SOURCE_DIR}/src/http/http_manager.cpp
)

target_include_directories(http PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external
)

add_library(ipc
  ${CMAKE_SOURCE_DIR}/src/ipc/ipc_manager.cpp
)

target_include_directories(ipc PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

add_library(tokeniser
  ${CMAKE_SOURCE_DIR}/src/tokeniser/json_tokeniser.cpp
  ${CMAKE_SOURCE_DIR}/src/formatter/formatter.cpp
)

target_include_directories(tokeniser PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external
)

add_library(engine
  ${CMAKE_SOURCE_DIR}/src/engine/engine_impl.cpp
  ${CMAKE_SOURCE_DIR}/src/engine/forward.cpp
)

target_include_directories(engine PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external
)

target_link_libraries(engine PUBLIC
  gguf
  tokeniser
)

add_library(common
  ${CMAKE_SOURCE_DIR}/src/stream_buffer.cpp
)

target_include_directories(common PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external
)

add_library(simdjson
  ${CMAKE_SOURCE_DIR}/external/simdjson/simdjson.cpp
)

target_include_directories(simdjson PUBLIC
  ${CMAKE_SOURCE_DIR}/external
)

add_library(application
  ${CMAKE_SOURCE_DIR}/src/app.cpp
)

target_include_directories(application PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/external
)

target_link_libraries(application PUBLIC
  odin_pch
  simdjson
  engine
  http
  ipc
  common
)
