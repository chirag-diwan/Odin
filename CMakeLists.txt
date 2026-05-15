cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(Odin)


add_subdirectory(./external/ggml)
add_executable(odin ./src/main.cpp)

target_include_directories(odin PRIVATE ./include )
target_link_libraries(odin PRIVATE ggml)

target_compile_options(odin PRIVATE
  -Wall
  -Wextra

  $<$<CONFIG:Debug>:-O0 -g3>
  $<$<CONFIG:Release>:-O3>
)
