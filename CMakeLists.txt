cmake_minimum_required(VERSION 3.16)

project(Odin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-g)

find_library(GGML_LIB ggml PATHS /usr/local/lib)
find_library(GGML_BASE_LIB ggml-base PATHS /usr/local/lib)
find_library(GGML_CPU_LIB ggml-cpu PATHS /usr/local/lib)
find_path(GGML_INCLUDE_DIR ggml.h PATHS /usr/local/include)

file(GLOB_RECURSE CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE C_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
)

add_executable(${PROJECT_NAME}
  ${C_SOURCES}
    ${CPP_SOURCES}
)

option(GGUF_DEBUG "Enable GGUF debug printing" OFF)
if(GGUF_DEBUG)
  target_compile_definitions(Odin PRIVATE GGUF_DEBUG)
endif()

option(MODEL_DEBUG "Enable Model debug" OFF)
if(MODEL_DEBUG)
  target_compile_definitions(Odin PRIVATE MODEL_DEBUG)
endif()


target_include_directories(${PROJECT_NAME} PRIVATE ${GGML_INCLUDE_DIR} src)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${GGML_LIB} 
    ${GGML_BASE_LIB} 
    ${GGML_CPU_LIB}
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
