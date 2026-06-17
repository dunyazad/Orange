cmake_minimum_required(VERSION 3.21)

project(Orange
    VERSION 0.1.0
    DESCRIPTION "ECS-driven, plugin-based rendering engine (OpenGL/Vulkan)"
    LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()

# All binaries (app + plugins) land in one directory so the app can discover
# plugins next to itself at runtime.
set(ORANGE_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${ORANGE_BIN_DIR}")  # .exe + .dll (Windows)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${ORANGE_BIN_DIR}")  # .so / .dylib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Build options ---------------------------------------------------------------
option(ORANGE_BUILD_GL "Build the OpenGL render plugin" ON)
option(ORANGE_BUILD_VK "Build the Vulkan render plugin (auto if SDK found)" ON)

# ---------------------------------------------------------------------------
# Dependencies (SDL3, EnTT) via FetchContent
# ---------------------------------------------------------------------------
include(cmake/dependencies.cmake)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_subdirectory(engine/render_api)   # interface-only contract
add_subdirectory(engine/core)         # platform + ECS

if(ORANGE_BUILD_GL)
    add_subdirectory(plugins/render_gl)
endif()

if(ORANGE_BUILD_VK)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Orange: Vulkan found (${Vulkan_VERSION}) -> building render_vk plugin")
        add_subdirectory(plugins/render_vk)
    else()
        message(STATUS "Orange: Vulkan SDK not found -> skipping render_vk plugin")
    endif()
endif()

add_subdirectory(app/sandbox)

message(STATUS "Orange: output dir = ${ORANGE_BIN_DIR}")
