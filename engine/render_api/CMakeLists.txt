# Header-only contract shared by the engine and every render plugin.
add_library(orange_render_api INTERFACE)
add_library(orange::render_api ALIAS orange_render_api)

target_include_directories(orange_render_api INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_features(orange_render_api INTERFACE cxx_std_17)
