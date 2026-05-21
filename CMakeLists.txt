cmake_minimum_required(VERSION 3.16)

project(Odin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(./external/ggml)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)



add_executable(odin
  ./src/main.cpp
)

target_include_directories(odin
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(odin PRIVATE ${PCRE2_INCLUDE_DIRS})
target_link_libraries(odin PRIVATE ${PCRE2_LIBRARIES})

target_link_libraries(odin
  PRIVATE
  ggml
)

target_compile_options(odin PRIVATE
  -Wall
  -Wextra
  $<$<CONFIG:Debug>:-O0 -g3>
  $<$<CONFIG:Release>:-O3>
)
