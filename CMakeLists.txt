cmake_minimum_required(VERSION 3.16)
project(Odin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXTERNAL_DIR ${CMAKE_SOURCE_DIR}/external)

add_subdirectory(./external/ggml)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED libpcre2-8)


option(ENABLE_TESTS "Build the test suite" OFF)
set(TEST_FILE "" CACHE FILEPATH "Path to test source file")


option(ENABLE_TESTS "Build the test suite" OFF)

if(ENABLE_TESTS)
  if(NOT TEST_FILE)
    message(FATAL_ERROR "ENABLE_TESTS is ON but TEST_FILE is not set")
  endif()

  get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)

  add_executable(${TEST_NAME} ${TEST_FILE})
  set(TARGET_NAME ${TEST_NAME})
else()
  add_executable(odin ./src/main.cpp)
  set(TARGET_NAME odin)
endif()

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PRIVATE ${PCRE2_INCLUDE_DIRS})

target_link_libraries(${TARGET_NAME}
  PRIVATE
  ggml
  ${PCRE2_LIBRARIES}
)

target_compile_options(${TARGET_NAME} PRIVATE
  -Wall
  -Wextra
  $<$<CONFIG:Debug>:-O0 -g3>
  $<$<CONFIG:Release>:-O3>
)
