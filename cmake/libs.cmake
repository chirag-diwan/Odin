add_library(odin_pch INTERFACE)

target_precompile_headers(odin_pch INTERFACE
  ${CMAKE_SOURCE_DIR}/external/httplib/httplib.h
  ${CMAKE_SOURCE_DIR}/external/nlohmann/json.hpp
  ${CMAKE_SOURCE_DIR}/external/jonathan/cpptui.hpp
)

add_library(gguf
  ${CMAKE_SOURCE_DIR}/src/gguf/gguf.cpp
  ${CMAKE_SOURCE_DIR}/src/gguf/ggufparser.cpp
)

add_library(http
  ${CMAKE_SOURCE_DIR}/src/http/http_manager.cpp
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
target_link_libraries(engine PUBLIC gguf tokeniser)

add_library(common
  ${CMAKE_SOURCE_DIR}/src/stream_buffer.cpp
)

add_library(simdjson
  ${CMAKE_SOURCE_DIR}/external/simdjson/simdjson.cpp
)

add_library(application
  ${CMAKE_SOURCE_DIR}/src/app.cpp
)

target_link_libraries(application PUBLIC 
  odin_pch
  simdjson
  engine 
  http 
  ipc 
  common
)
