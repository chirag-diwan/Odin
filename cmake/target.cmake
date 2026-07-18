function(add_odin_target target_name source_file)
  add_executable(${target_name} 
    ${source_file}
  )

  target_precompile_headers(${target_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/external/httplib/httplib.h
  )

  target_include_directories(${target_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/simdjson
    ${PCRE2_INCLUDE_DIRS}
  )


  target_link_libraries(${target_name} PRIVATE
    simdjson
    engine 
    http 
    ipc 
    common
    ggml
    replxx
    ${PCRE2_LIBRARIES}
  )

  target_compile_options(${target_name} PRIVATE
    -Wall
    -Wextra
    $<$<CONFIG:Debug>:-O0 -ggdb -pthread>
    $<$<CONFIG:Release>:-O3 -march=native>
  )
endfunction()
