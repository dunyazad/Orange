# Core engine: platform layer (SDL3), plugin loader, the EnTT-based ECS, and the
# CPU geometry toolkit (sparse grid, TSDF voxels, surface extraction).
add_library(orange_core STATIC
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

target_include_directories(orange_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")  # robin_hood (header-only)
target_include_directories(orange_core PRIVATE ${STB_INCLUDE_DIR})  # stb_image_write

# Eigen backs orange/core/math.h (a public header), so it must be on the public
# interface. EIGEN_MAX_ALIGN_BYTES=0 disables over-alignment so Eigen members are
# safe inside EnTT component pools (std::allocator-backed).
if(EIGEN_INCLUDE_DIR)
    target_include_directories(orange_core SYSTEM PUBLIC "${EIGEN_INCLUDE_DIR}")
endif()
target_compile_definitions(orange_core PUBLIC EIGEN_MAX_ALIGN_BYTES=0)

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
    target_link_libraries(orange_core PUBLIC "${ORANGE_ONNX_DIR}/lib/onnxruntime.lib")
    target_compile_definitions(orange_core PUBLIC ORANGE_ENABLE_ONNX=1
        ORANGE_ONNX_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/onnx/models")
    # Copy the runtime DLL next to every consumer exe.
    file(COPY "${ORANGE_ONNX_DIR}/lib/onnxruntime.dll" DESTINATION "${ORANGE_BIN_DIR}")
    if(DEFINED CMAKE_CONFIGURATION_TYPES)
        foreach(cfg ${CMAKE_CONFIGURATION_TYPES})
            file(COPY "${ORANGE_ONNX_DIR}/lib/onnxruntime.dll" DESTINATION "${ORANGE_BIN_DIR}/${cfg}")
        endforeach()
    endif()
else()
    message(STATUS "Orange: ONNX Runtime not found -> MobileSAM segmentation stubbed")
endif()

target_compile_features(orange_core PUBLIC cxx_std_17)
