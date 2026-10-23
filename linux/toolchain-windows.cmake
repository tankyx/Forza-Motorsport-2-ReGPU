# Windows cross-build from Linux: clang-cl + an xwin sysroot + lld-link.
#
#   export INCLUDE="<sysroot>/crt/include;<sysroot>/sdk/include/{ucrt,um,shared,winrt}"
#   export LIB="<sysroot>/crt/lib/x86_64;<sysroot>/sdk/lib/{um,ucrt}/x86_64"
#   cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=linux/toolchain-windows.cmake
#
# linux/build.sh does all of that; this file is the part CMake needs.
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_RC_COMPILER llvm-rc)
set(CMAKE_LINKER lld-link)
set(CMAKE_AR llvm-lib)
set(CMAKE_C_FLAGS_INIT "-fuse-ld=lld-link")
set(CMAKE_CXX_FLAGS_INIT "-fuse-ld=lld-link /EHsc")

# xwin's splat has no debug CRT (msvcrtd.lib), so keep CMake's compiler checks on the
# release runtime instead of the Debug defaults.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_C_FLAGS_DEBUG "-Ob0 -Od")
set(CMAKE_CXX_FLAGS_DEBUG "-Ob0 -Od")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "")
