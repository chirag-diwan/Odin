function(download_file URL OUTPUT_FILE)
  if(EXISTS "${OUTPUT_FILE}")
    return()
  endif()

  get_filename_component(FILE_NAME "${OUTPUT_FILE}" NAME)
  get_filename_component(OUTPUT_DIR "${OUTPUT_FILE}" DIRECTORY)

  message(STATUS "[DOWNLOAD] ${FILE_NAME}")
  message(STATUS "[SOURCE]   ${URL}")
  message(STATUS "[TARGET]   ${OUTPUT_FILE}")

  file(MAKE_DIRECTORY "${OUTPUT_DIR}")

  file(
    DOWNLOAD
    "${URL}"
    "${OUTPUT_FILE}"
    SHOW_PROGRESS
    STATUS DOWNLOAD_STATUS
  )

  list(GET DOWNLOAD_STATUS 0 STATUS_CODE)
  list(GET DOWNLOAD_STATUS 1 STATUS_MESSAGE)

  if(NOT STATUS_CODE EQUAL 0)
    message(FATAL_ERROR
      "\n"
      "[DOWNLOAD FAILED]\n"
      "File   : ${FILE_NAME}\n"
      "Source : ${URL}\n"
      "Reason : ${STATUS_MESSAGE}\n"
    )
  endif()
endfunction()
