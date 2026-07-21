download_file(
  "https://raw.githubusercontent.com/simdjson/simdjson/refs/heads/master/singleheader/simdjson.h"
  "${CMAKE_SOURCE_DIR}/external/simdjson/simdjson.h"
)


download_file(
  "https://raw.githubusercontent.com/simdjson/simdjson/refs/heads/master/singleheader/simdjson.cpp"
  "${CMAKE_SOURCE_DIR}/external/simdjson/simdjson.cpp"
)

download_file(
  "https://raw.githubusercontent.com/yhirose/cpp-httplib/refs/heads/master/httplib.h"
  "${CMAKE_SOURCE_DIR}/external/httplib/httplib.h"
)

download_archive(
  "https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz"
  "${CMAKE_SOURCE_DIR}/external/nlohmann/source.tar.gz"
  "${CMAKE_SOURCE_DIR}/external/nlohmann/"
)
