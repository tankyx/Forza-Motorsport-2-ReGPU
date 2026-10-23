# Included with CMAKE_PROJECT_TOP_LEVEL_INCLUDES, i.e. right after project().
#
# Cross-compiling with clang-cl from Linux, CMake records no C++ compile features at all
# (CMAKE_CXX20_COMPILE_FEATURES ends up empty), and then every CXX_STANDARD request —
# which this project's CMakeLists and fmt both make — is a generate-time error:
#   "No known features for CXX compiler "Clang" version ..."
# The standard levels are what the targets ask for, so supply them.
set(CMAKE_CXX_COMPILE_FEATURES cxx_std_98 cxx_std_11 cxx_std_14 cxx_std_17 cxx_std_20 cxx_std_23 cxx_std_26)
foreach(std 98 11 14 17 20 23 26)
  set(CMAKE_CXX${std}_COMPILE_FEATURES cxx_std_${std})
endforeach()
set(CMAKE_CXX_KNOWN_FEATURES ${CMAKE_CXX_COMPILE_FEATURES})
