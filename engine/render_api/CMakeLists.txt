# Header-only contract shared by the engine and every render plugin.
add_library(orange_render_api INTERFACE)
add_library(orange::render_api ALIAS orange_render_api)

# BUILD_INTERFACE / INSTALL_INTERFACE so the exported package points at the
# installed header tree instead of this source dir (see cmake/OrangeInstall.cmake).
target_include_directories(orange_render_api INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

target_compile_features(orange_render_api INTERFACE cxx_std_17)
