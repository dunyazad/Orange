# Vulkan render backend, built as a hot-swappable shared library (plugin).
# Only configured when find_package(Vulkan) succeeds (see top-level CMakeLists).
add_library(render_vk MODULE
    src/vk_renderer.cpp
    src/plugin_main.cpp)

target_link_libraries(render_vk
    PRIVATE
        orange::render_api
        SDL3::SDL3
        Vulkan::Vulkan)

set_target_properties(render_vk PROPERTIES OUTPUT_NAME "render_vk")

target_compile_features(render_vk PRIVATE cxx_std_17)

# Regenerate the embedded SPIR-V headers from GLSL when the shaders change.
# The committed headers in src/generated/ are used as-is if glslangValidator is
# unavailable, so the build never hard-depends on it.
set(_shader_gen "${CMAKE_CURRENT_SOURCE_DIR}/src/generated")
if(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
    add_custom_command(
        OUTPUT "${_shader_gen}/mesh.vert.spv.h"
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh.vert"
                --vn g_meshVertSpv -o "${_shader_gen}/mesh.vert.spv.h"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh.vert"
        COMMENT "Compiling mesh.vert -> SPIR-V header"
        VERBATIM)
    add_custom_command(
        OUTPUT "${_shader_gen}/mesh.frag.spv.h"
        COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh.frag"
                --vn g_meshFragSpv -o "${_shader_gen}/mesh.frag.spv.h"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh.frag"
        COMMENT "Compiling mesh.frag -> SPIR-V header"
        VERBATIM)
    target_sources(render_vk PRIVATE
        "${_shader_gen}/mesh.vert.spv.h"
        "${_shader_gen}/mesh.frag.spv.h")
endif()
