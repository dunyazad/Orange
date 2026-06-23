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
    src/geometry/normals.cpp
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

target_compile_features(orange_core PUBLIC cxx_std_17)
