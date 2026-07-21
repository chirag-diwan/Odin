function(download_archive URL DEST_ARCHIVE DEST_DIR)
  if(EXISTS "${DEST_ARCHIVE}")
    return()
  endif()
  message(STATUS "[DOWNLOAD] ${DEST_DIR}")
  message(STATUS "[SOURCE]   ${URL}")
  message(STATUS "[TARGET]   ${DEST_ARCHIVE}")

  file(DOWNLOAD
    "${URL}"
    "${DEST_ARCHIVE}"
    SHOW_PROGRESS
  )

  file(MAKE_DIRECTORY "${DEST_DIR}")

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf "${DEST_ARCHIVE}"
    WORKING_DIRECTORY "${DEST_DIR}"
    RESULT_VARIABLE result
  )

  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Failed to extract ${DEST_ARCHIVE}")
  endif()


  file(GLOB children LIST_DIRECTORIES true "${DEST_DIR}/*")

  foreach(child ${children})
    if(IS_DIRECTORY "${child}")
      file(GLOB contents "${child}/*")
      file(COPY ${contents} DESTINATION "${DEST_DIR}")
      file(REMOVE_RECURSE "${child}")
      break()
    endif()
  endforeach()
endfunction()
