# ---------------------------------------------------------------------------
# Install / export rules -- turn the engine into a redistributable binary SDK.
#
# Goal: another project consumes Orange as *binaries + headers*, never as
# source. After
#
#     cmake --install build --config Release --prefix C:/SDK/Orange
#
# the prefix holds everything a consumer needs:
#
#   <prefix>/include/orange/core/*.h         orange_core public headers (C++)
#   <prefix>/include/orange/c/orange.h       the C ABI facade (toolchain-free)
#   <prefix>/include/orange/render/*.h       the C-ABI render contract
#   <prefix>/include/orange-third_party/     bundled header-only deps
#                     robin_hood/            (robin_hood, Eigen) so the
#                     eigen/Eigen            consumer needs no extra installs
#   <prefix>/lib/orange_core.lib             the static core (orange_cored.lib in Debug)
#   <prefix>/lib/cmake/Orange/*.cmake        find_package(Orange) package config
#   <prefix>/lib/cmake/{SDL3,EnTT}/          deps re-exported into the same prefix
#   <prefix>/bin/orange_c.dll                the C ABI facade (self-contained)
#   <prefix>/bin/render_gl.dll               runtime-loaded render plugins
#   <prefix>/bin/render_vk.dll               (+ SDL3.dll, onnxruntime.dll)
#
# and the consumer side is:
#
#     find_package(Orange CONFIG REQUIRED)
#     add_executable(myApp main.cpp)
#     target_link_libraries(myApp PRIVATE orange::core)
#     orange_copy_runtime(myApp)   # plugins + SDL3 next to the exe
#
# Included from the top-level CMakeLists when ORANGE_INSTALL is ON.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(ORANGE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/Orange")

# --- exported target names --------------------------------------------------
# Match the in-tree aliases, so consumer code is identical whether Orange is
# add_subdirectory()'d or found as a package: orange::core / orange::render_api.
set_target_properties(orange_core       PROPERTIES EXPORT_NAME core)
set_target_properties(orange_render_api PROPERTIES EXPORT_NAME render_api)
if(TARGET orange_c)
    set_target_properties(orange_c PROPERTIES EXPORT_NAME c)
endif()

# Debug and Release static libs can then coexist in one prefix; the generated
# OrangeTargets-<config>.cmake picks the right one per consumer config.
set_target_properties(orange_core PROPERTIES DEBUG_POSTFIX d)

# --- 1. libraries -----------------------------------------------------------
set(ORANGE_EXPORTED_TARGETS orange_core orange_render_api)
if(TARGET orange_c)
    list(APPEND ORANGE_EXPORTED_TARGETS orange_c)
endif()

install(TARGETS ${ORANGE_EXPORTED_TARGETS}
        EXPORT  OrangeTargets
        ARCHIVE  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        LIBRARY  DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME  DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# --- 2. render plugins ------------------------------------------------------
# MODULE libraries: never linked, dlopen'd at runtime from the directory next
# to the executable, so they install into bin/ and get copied by
# orange_copy_runtime(). They are not part of the export set (a MODULE has no
# link interface to export) -- the package exposes them as ORANGE_PLUGIN_DIR.
foreach(_plugin render_gl render_vk)
    if(TARGET ${_plugin})
        install(TARGETS ${_plugin}
                LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}"
                RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
endforeach()

# --- 3. public headers ------------------------------------------------------
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engine/core/include/orange"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engine/render_api/include/orange"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.h")

# The C ABI header is self-contained (stdint.h only) -- a consumer using
# orange_c.dll needs nothing else from include/.
if(TARGET orange_c)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engine/c_api/include/orange"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
            FILES_MATCHING PATTERN "*.h")
endif()

# Header-only third-party that leaks into public headers, bundled so the
# consumer does not have to fetch anything:
#   robin_hood -> <robin_hood/robin_hood.h> from orange/core/sparse_grid.h
#   Eigen      -> <Eigen/...>               from orange/core/math.h
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/engine/third_party/robin_hood"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/orange-third_party")

# Headers only -- Eigen's checkout also carries its tests/CMake, which would
# add ~100 MB of noise to the SDK.
if(EIGEN_INCLUDE_DIR AND NOT TARGET Eigen3::Eigen)
    install(DIRECTORY "${EIGEN_INCLUDE_DIR}/Eigen"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/orange-third_party/eigen"
            PATTERN "CMakeLists.txt" EXCLUDE)
    if(EXISTS "${EIGEN_INCLUDE_DIR}/unsupported/Eigen")
        install(DIRECTORY "${EIGEN_INCLUDE_DIR}/unsupported/Eigen"
                DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/orange-third_party/eigen/unsupported"
                PATTERN "CMakeLists.txt" EXCLUDE)
    endif()
endif()

# --- 4. optional ONNX Runtime ----------------------------------------------
# Kept out of the exported link interface (absolute build-machine paths are not
# relocatable); OrangeConfig.cmake re-attaches the installed import library.
if(ORANGE_ONNX_RUNTIME_DIR)
    install(FILES "${ORANGE_ONNX_RUNTIME_DIR}/lib/onnxruntime.lib"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}")
    install(FILES "${ORANGE_ONNX_RUNTIME_DIR}/lib/onnxruntime.dll"
            DESTINATION "${CMAKE_INSTALL_BINDIR}")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/models")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnx/models/"
                DESTINATION "${CMAKE_INSTALL_DATADIR}/orange/models")
    endif()
endif()

# --- 5. the package itself --------------------------------------------------
# Does Orange link a real Eigen3::Eigen imported target (system Eigen)? Then the
# consumer must find_package(Eigen3) too; otherwise the bundled headers are used.
if(TARGET Eigen3::Eigen)
    set(ORANGE_EXPORT_EIGEN_PACKAGE 1)
else()
    set(ORANGE_EXPORT_EIGEN_PACKAGE 0)
endif()

install(EXPORT OrangeTargets
        FILE        OrangeTargets.cmake
        NAMESPACE   orange::
        DESTINATION "${ORANGE_INSTALL_CMAKEDIR}")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OrangeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OrangeConfig.cmake"
    INSTALL_DESTINATION "${ORANGE_INSTALL_CMAKEDIR}"
    PATH_VARS CMAKE_INSTALL_BINDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/OrangeConfigVersion.cmake"
    VERSION       "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion)

install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/OrangeConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/OrangeConfigVersion.cmake"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OrangeRuntime.cmake"
        DESTINATION "${ORANGE_INSTALL_CMAKEDIR}")

message(STATUS "Orange: install/export enabled (package dir = ${ORANGE_INSTALL_CMAKEDIR})")
