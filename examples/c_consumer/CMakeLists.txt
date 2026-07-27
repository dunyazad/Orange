# ---------------------------------------------------------------------------
# Example: consuming Orange through the C ABI (orange_c.dll).
#
# Standalone against an installed prefix, exactly like examples/consumer, but
# the language here is C -- the point being that the compiler does not have to
# match the one that built the engine:
#
#   cmake -S examples/c_consumer -B build-c -G "Visual Studio 17 2022" -A x64 ^
#         -DCMAKE_PREFIX_PATH=C:/SDK/Orange
#   cmake --build build-c --config Release
#   build-c/Release/orange_c_consumer.exe
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)
project(orange_c_consumer LANGUAGES C)

find_package(Orange CONFIG REQUIRED)

add_executable(orange_c_consumer main.c)
target_link_libraries(orange_c_consumer PRIVATE orange::c)

# Only orange_c.dll is actually needed here (it statically contains the core
# unless the SDK was built with -Shared); orange_copy_runtime() drops the whole
# bin/ next to the exe, which also covers that case.
orange_copy_runtime(orange_c_consumer)
