# ---------------------------------------------------------------------------
# Example: a project that consumes Orange as a *binary* SDK.
#
# This directory is NOT part of the Orange build -- configure it standalone
# against an installed prefix:
#
#   cmake -S examples/consumer -B build-consumer -G "Visual Studio 17 2022" -A x64 ^
#         -DCMAKE_PREFIX_PATH=C:/SDK/Orange
#   cmake --build build-consumer --config Release
#   build-consumer/Release/orange_consumer.exe
#
# Note the total absence of Orange source: only find_package + link.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)
project(orange_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Orange CONFIG REQUIRED)

# --- headless tool: geometry/IO toolkit only, no window, no renderer --------
add_executable(orange_consumer main.cpp)
target_link_libraries(orange_consumer PRIVATE orange::core)

# orange::core links SDL3 (platform layer), so even a headless tool needs
# SDL3.dll beside it; this also drops the render plugins there, which is what a
# windowed app loads at runtime.
orange_copy_runtime(orange_consumer)

# --- full app: executable + link + runtime copy in one call -----------------
# orange_add_app(myViewer SOURCES viewer.cpp)

message(STATUS "Orange ${Orange_VERSION} from ${ORANGE_INCLUDE_DIR} (plugins: ${ORANGE_PLUGIN_DIR})")
