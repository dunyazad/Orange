# Core engine: platform layer (SDL3), plugin loader, and the EnTT-based ECS.
add_library(orange_core STATIC
    src/plugin_loader.cpp
    src/window.cpp
    src/systems.cpp
    src/application.cpp
    src/screenshot.cpp
    src/console.cpp)
add_library(orange::core ALIAS orange_core)

target_include_directories(orange_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
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

target_compile_features(orange_core PUBLIC cxx_std_17)
