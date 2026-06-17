# OpenGL render backend, built as a hot-swappable shared library (plugin).
add_library(render_gl MODULE
    src/gl_loader.cpp
    src/gl_renderer.cpp
    src/plugin_main.cpp)

target_link_libraries(render_gl
    PRIVATE
        orange::render_api
        SDL3::SDL3)

# Keep default platform prefixing: render_gl.dll on Windows, librender_gl.so
# on Unix, librender_gl.dylib on macOS -- exactly what pluginFileName() builds.
set_target_properties(render_gl PROPERTIES OUTPUT_NAME "render_gl")

target_compile_features(render_gl PRIVATE cxx_std_17)
