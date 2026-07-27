# Core engine: platform layer (SDL3), plugin loader, the EnTT-based ECS, and the
# CPU geometry toolkit (sparse grid, TSDF voxels, surface extraction).
#
# STATIC by default; ORANGE_BUILD_SHARED=ON builds it as orange_core.dll /
# liborange_core.so instead, so a consuming project ships one shared binary
# instead of linking the whole archive (see cmake/OrangeInstall.cmake and
# docs/USING_AS_LIBRARY.md). The interface is still C++ either way -- the same
# compiler + runtime is required on both sides.
if(ORANGE_BUILD_SHARED)
    set(_orange_core_type SHARED)
else()
    set(_orange_core_type STATIC)
endif()

add_library(orange_core ${_orange_core_type}
    src/plugin_loader.cpp
    src/window.cpp
    src/systems.cpp
    src/application.cpp
    src/screenshot.cpp
    src/console.cpp
    src/crash_handler.cpp
    src/ui_layout.cpp
    src/geometry/sparse_grid.cpp
    src/geometry/sparse_data_block.cpp
    src/geometry/mesh_generation.cpp
    src/geometry/poisson_reconstruction.cpp
    src/geometry/half_edge.cpp
    src/geometry/normals.cpp
    src/geometry/occlusal_plane.cpp
    src/geometry/occlusal_render.cpp
    src/geometry/tooth_segment.cpp
    src/ml/sam_segment.cpp
    src/geometry/primitives.cpp
    src/geometry/point_ops.cpp
    src/io/serialization.cpp
    src/modes/modes.cpp)
add_library(orange::core ALIAS orange_core)

# Shared build: no __declspec(dllexport) annotations exist in the headers, so
# let CMake generate the .def from the object files (functions + methods; there
# are no exported data globals -- DebugDraw is a function-local static). -fPIC
# on Unix-likes.
set_target_properties(orange_core PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    POSITION_INDEPENDENT_CODE  ON)

if(ORANGE_BUILD_SHARED)
    set_target_properties(orange_core PROPERTIES
        VERSION   "${PROJECT_VERSION}"
        SOVERSION "${PROJECT_VERSION_MAJOR}")
endif()

# Public headers, expressed with BUILD_INTERFACE / INSTALL_INTERFACE so the same
# target works from this build tree AND from an installed binary SDK, where the
# headers live under <prefix>/include (see cmake/OrangeInstall.cmake).
target_include_directories(orange_core PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
    # robin_hood (header-only) -- leaks into sparse_grid.h, so it ships too.
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../third_party>"
    "$<INSTALL_INTERFACE:include/orange-third_party>")
target_include_directories(orange_core PRIVATE ${STB_INCLUDE_DIR})  # stb_image_write

# Eigen backs orange/core/math.h (a public header), so it must be on the public
# interface. EIGEN_MAX_ALIGN_BYTES=0 disables over-alignment so Eigen members are
# safe inside EnTT component pools (std::allocator-backed).
# The fetched (non-package) Eigen is bundled into the SDK under
# include/orange-third_party/eigen, so an installed consumer needs no Eigen of
# its own; a system Eigen3::Eigen (linked below) carries its own include dirs.
if(EIGEN_INCLUDE_DIR)
    target_include_directories(orange_core SYSTEM PUBLIC
        "$<BUILD_INTERFACE:${EIGEN_INCLUDE_DIR}>"
        "$<INSTALL_INTERFACE:include/orange-third_party/eigen>")
endif()
target_compile_definitions(orange_core PUBLIC EIGEN_MAX_ALIGN_BYTES=0)

# systems.cpp (many EnTT/Eigen template instantiations) exceeds the default
# COFF section limit in Debug builds; /bigobj lifts it.
if(MSVC)
    target_compile_options(orange_core PRIVATE /bigobj)
endif()

target_link_libraries(orange_core
    PUBLIC
        orange::render_api
        EnTT::EnTT
        SDL3::SDL3)
if(TARGET Eigen3::Eigen)
    target_link_libraries(orange_core PUBLIC Eigen3::Eigen)
endif()

# Needed for dlopen/dlsym on Unix-likes.
if(NOT WIN32)
    target_link_libraries(orange_core PUBLIC ${CMAKE_DL_LIBS})
endif()

# DbgHelp backs the crash handler's stack walk + minidump on Windows.
if(WIN32)
    target_link_libraries(orange_core PUBLIC dbghelp)
endif()

# ONNX Runtime (optional): enables the MobileSAM 2D segmentation. Auto-detected
# from the vendored prebuilt; sam_segment.cpp is a no-op stub when absent.
set(ORANGE_ONNX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnx/onnxruntime-win-x64-1.27.0")
if(WIN32 AND EXISTS "${ORANGE_ONNX_DIR}/lib/onnxruntime.lib")
    message(STATUS "Orange: ONNX Runtime found -> MobileSAM segmentation enabled")
    target_include_directories(orange_core PRIVATE "${ORANGE_ONNX_DIR}/include")
    # BUILD_INTERFACE-only: an absolute path on this machine must not end up in
    # the installed package. OrangeConfig.cmake re-attaches the SDK's own copy.
    target_link_libraries(orange_core PUBLIC
        "$<BUILD_INTERFACE:${ORANGE_ONNX_DIR}/lib/onnxruntime.lib>")
    # Only sam_segment.cpp reads these, so they stay off the public interface.
    target_compile_definitions(orange_core PRIVATE ORANGE_ENABLE_ONNX=1
        ORANGE_ONNX_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnx/models")
    # Let the packaging step pick up the runtime + models.
    set(ORANGE_ONNX_RUNTIME_DIR "${ORANGE_ONNX_DIR}" CACHE INTERNAL "ONNX Runtime dir")
    # Copy the runtime DLL next to every consumer exe.
    file(COPY "${ORANGE_ONNX_DIR}/lib/onnxruntime.dll" DESTINATION "${ORANGE_BIN_DIR}")
    if(DEFINED CMAKE_CONFIGURATION_TYPES)
        foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
            file(COPY "${ORANGE_ONNX_DIR}/lib/onnxruntime.dll" DESTINATION "${ORANGE_BIN_DIR}/${cfg}")
        endforeach()
    endif()
else()
    message(STATUS "Orange: ONNX Runtime not found -> MobileSAM segmentation stubbed")
    unset(ORANGE_ONNX_RUNTIME_DIR CACHE)
endif()

target_compile_features(orange_core PUBLIC cxx_std_17)
