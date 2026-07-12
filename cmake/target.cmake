function(add_odin_target target_name source_file)
  add_executable(${target_name} ${source_file})

  target_precompile_headers(${target_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/external/httplib/httplib.h
  )

  target_include_directories(${target_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${PCRE2_INCLUDE_DIRS}
    ${EXTERNAL_DIR}/simdjson
  )

  target_sources(${target_name} PRIVATE
    ${EXTERNAL_DIR}/simdjson/simdjson.cpp
  )

  target_link_libraries(${target_name} PRIVATE
    ggml
    replxx
    ${PCRE2_LIBRARIES}
  )

  target_compile_options(${target_name} PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-O0 -g3 -pthread>
    $<$<CONFIG:Release>:-O3>
  )
endfunction()
