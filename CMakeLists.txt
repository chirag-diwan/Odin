cmake_minimum_required(VERSION 3.16)
project(Odin)

include("cmake/downloadFile.cmake")
include("cmake/downloads.cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)

add_subdirectory("${EXTERNAL_DIR}/ggml")
add_subdirectory("${EXTERNAL_DIR}/minja")

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)

option(ENABLE_TESTS "Build the test suite" OFF)
set(TEST_FILE "" CACHE FILEPATH "Path to test source file")


if(ENABLE_TESTS)
  if(NOT TEST_FILE)
    message(FATAL_ERROR "ENABLE_TESTS is ON but TEST_FILE is not set")
  endif()
  get_filename_component(TEST_BASENAME ${TEST_FILE} NAME_WE)
  add_executable(${TEST_BASENAME}_TEST ${TEST_FILE})
  set(TARGET_NAME ${TEST_BASENAME}_TEST)
else()
  add_executable(odin ./src/engine.cpp)
  set(TARGET_NAME odin)
endif()

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PRIVATE ${PCRE2_INCLUDE_DIRS})

target_include_directories(${TARGET_NAME}
    PRIVATE
    ${EXTERNAL_DIR}/simdjson
)

target_sources(${TARGET_NAME}
    PRIVATE
    ${EXTERNAL_DIR}/simdjson/simdjson.cpp
)

target_link_libraries(${TARGET_NAME}
  PRIVATE
  ggml
  minja
  ${PCRE2_LIBRARIES}
)

target_compile_options(${TARGET_NAME} PRIVATE
  -Wall
  -Wextra
  $<$<CONFIG:Debug>:-O0 -g3>
  $<$<CONFIG:Release>:-O3>
)
