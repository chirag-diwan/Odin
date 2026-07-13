cmake_minimum_required(VERSION 3.16)
project(Odin)

include("cmake/target.cmake")
include("cmake/downloadFile.cmake")
include("cmake/downloads.cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

list(REMOVE_ITEM COMMON_SOURCES
  "${CMAKE_SOURCE_DIR}/src/engine.cpp"
  "${CMAKE_SOURCE_DIR}/src/http-server.cpp"
)

add_library(common ${COMMON_SOURCES})

add_subdirectory("${EXTERNAL_DIR}/ggml")
add_subdirectory("${EXTERNAL_DIR}/replxx")

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

option(ENABLE_TESTS "Build the test suite" OFF)
set(TEST_FILE "" CACHE FILEPATH "Path to test source file")

if(ENABLE_TESTS)
  if(NOT TEST_FILE)
    message(FATAL_ERROR "ENABLE_TESTS is ON but TEST_FILE is not set")
  endif()

  get_filename_component(TEST_BASENAME ${TEST_FILE} NAME_WE)
  add_odin_target(${TEST_BASENAME}_TEST ${TEST_FILE})
else()
  add_odin_target(odin ./src/engine.cpp)
  add_odin_target(odin-http-server ./src/http-server.cpp)
endif()
