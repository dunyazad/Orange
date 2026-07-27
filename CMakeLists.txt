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
option(ORANGE_BUILD_TESTS "Build the headless orange_core unit tests" ON)
# Install/export rules: produce a redistributable binary SDK (headers + libs +
# plugin DLLs + a find_package(Orange) package config). See cmake/OrangeInstall.cmake.
option(ORANGE_INSTALL "Generate install/export rules for the binary SDK" ON)
# Ship orange_core as a DLL/.so instead of a static archive. Consumers link the
# small import library and load orange_core.dll at runtime (copied next to the
# exe by orange_copy_runtime()).
option(ORANGE_BUILD_SHARED "Build orange_core as a shared library (DLL)" OFF)
# orange_c.dll: a plain-C facade over the CPU toolkit. Unlike orange_core it is
# toolchain-independent (no C++ types in the interface), so any compiler or
# language with an FFI can consume it. See engine/c_api/include/orange/c/orange.h.
option(ORANGE_BUILD_C_API "Build the C ABI wrapper (orange_c)" ON)
# Link the MSVC runtime statically (/MT). Then orange_c.dll imports nothing but
# KERNEL32 -- no VC++ redistributable on the target machine, and no chance of a
# CRT mismatch with whatever built the consumer. Applies to the whole build
# (mixing /MT and /MD in one link is not allowed).
option(ORANGE_STATIC_CRT "Link the MSVC runtime statically (/MT)" OFF)

if(ORANGE_STATIC_CRT AND MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)  # honour the above in subprojects too
endif()
option(ORANGE_INSTALL_APPS "Also install the demo apps into the SDK prefix" OFF)

if(ORANGE_BUILD_TESTS)
    enable_testing()  # so `ctest --test-dir build` finds the tests below
endif()

# ---------------------------------------------------------------------------
# Dependencies (SDL3, EnTT) via FetchContent
# ---------------------------------------------------------------------------
include(cmake/dependencies.cmake)

# Treat our sources as UTF-8 on MSVC. Without this, MSVC assumes the system code
# page (e.g. 949 on Korean Windows) and warns C4819 on third-party headers (EnTT)
# that contain UTF-8 characters. Applied here so it covers the engine + apps but
# not the FetchContent dependency builds above.
if(MSVC)
    add_compile_options(/utf-8)
    # Keep debug symbols in Release too, so the crash handler (and Visual Studio /
    # WinDbg on the minidump) can resolve file:line in optimized builds. /Zi emits
    # a PDB; /DEBUG keeps it at link time; /OPT:REF,ICF undo the size cost /DEBUG
    # would otherwise leave so the Release binary stays lean.
    add_compile_options($<$<CONFIG:Release>:/Zi>)
    add_link_options($<$<CONFIG:Release>:/DEBUG>
                     $<$<CONFIG:Release>:/OPT:REF>
                     $<$<CONFIG:Release>:/OPT:ICF>)
endif()

# ---------------------------------------------------------------------------
# Targets
#
# Two layers: the engine (reusable core + render plugins), then the apps that
# sit on top of it. Adding a new app is a new add_subdirectory below + a thin
# orange_add_app() in that subdir -- the engine never changes.
# ---------------------------------------------------------------------------
add_subdirectory(engine)              # the reusable engine (core + render plugins)

# Apps built on the engine. Add more here, each a thin orange_add_app().
add_subdirectory(app/appOrange)
add_subdirectory(app/cusp_probe)      # CLI diagnostic for findCusps tuning

# ---------------------------------------------------------------------------
# Packaging: install + export the engine so other projects consume it as
# binaries (no source drop-in). Last, so every target above exists.
# ---------------------------------------------------------------------------
if(ORANGE_INSTALL)
    include(cmake/OrangeInstall.cmake)
    if(ORANGE_INSTALL_APPS)
        install(TARGETS appOrange cusp_probe RUNTIME DESTINATION bin)
    endif()
endif()

message(STATUS "Orange: output dir = ${ORANGE_BIN_DIR}")
