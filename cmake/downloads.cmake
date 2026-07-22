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

download_file(
  "https://raw.githubusercontent.com/nlohmann/json/refs/heads/develop/single_include/nlohmann/json.hpp"
  "${CMAKE_SOURCE_DIR}/external/nlohmann/json.hpp"
)

download_file(
  "https://raw.githubusercontent.com/nlohmann/json/refs/heads/develop/single_include/nlohmann/json_fwd.hpp"
  "${CMAKE_SOURCE_DIR}/external/nlohmann/json_fwd.hpp"
)
