# ===========================================================================
# The Orange engine.
#
# This is the reusable, app-agnostic half of the project: the C-ABI render
# contract, the ECS + platform core, and the render-backend plugins the engine
# ships. Apps sit on top of it via orange_add_app() (see cmake/OrangeApp.cmake)
# and talk only to orange::core -- never to GL/VK/SDL directly.
#
# Anything an application needs to depend on is defined here; nothing in this
# tree depends on any app, so multiple apps can be built against one engine.
# ===========================================================================

# orange_add_app(): the entry point apps use. Defined here (CMake functions are
# global once the defining add_subdirectory has run) so every app subdirectory
# added after engine/ can call it.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/OrangeApp.cmake")

add_subdirectory(render_api)   # interface-only C-ABI contract
add_subdirectory(core)         # platform layer + ECS + plugin loader

# Render backends, built as hot-swappable plugins that land in the shared output
# dir next to the app. ORANGE_BUILD_GL/VK are defined at the top level.
if(ORANGE_BUILD_GL)
    add_subdirectory(plugins/render_gl)
endif()

if(ORANGE_BUILD_VK)
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        message(STATUS "Orange: Vulkan found (${Vulkan_VERSION}) -> building render_vk plugin")
        add_subdirectory(plugins/render_vk)
    else()
        message(STATUS "Orange: Vulkan SDK not found -> skipping render_vk plugin")
    endif()
endif()
