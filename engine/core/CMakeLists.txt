# Core engine: platform layer (SDL3), plugin loader, and the EnTT-based ECS.
add_library(orange_core STATIC
    src/plugin_loader.cpp
    src/window.cpp
    src/systems.cpp
    src/application.cpp
    src/screenshot.cpp)
add_library(orange::core ALIAS orange_core)

target_include_directories(orange_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(orange_core PRIVATE ${STB_INCLUDE_DIR})  # stb_image_write

target_link_libraries(orange_core
    PUBLIC
        orange::render_api
        EnTT::EnTT
        SDL3::SDL3)

# Needed for dlopen/dlsym on Unix-likes.
if(NOT WIN32)
    target_link_libraries(orange_core PUBLIC ${CMAKE_DL_LIBS})
endif()

target_compile_features(orange_core PUBLIC cxx_std_17)
