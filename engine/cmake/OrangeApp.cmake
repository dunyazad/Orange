# orange_add_app(<name> SOURCES <src>...)
#
# Declares an application built on top of the Orange engine. The app links only
# the engine core (orange::core) + SDL3 -- never GL/VK directly -- and discovers
# the render-backend plugins next to itself at runtime. This helper wires up the
# bits every app needs so adding a new app is a one-liner:
#
#     orange_add_app(myApp SOURCES src/main.cpp)
#
#   * links orange::core + SDL3 and requires C++17
#   * makes the render plugins build alongside the app (same output dir)
#   * on Windows, copies SDL3.dll next to the executable so it runs in place
#
# App-specific extras (e.g. stb include dirs, extra sources) are added by the
# caller on the returned target by its <name>.
function(orange_add_app NAME)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    if(NOT APP_SOURCES)
        message(FATAL_ERROR "orange_add_app(${NAME}): no SOURCES given")
    endif()

    add_executable(${NAME} ${APP_SOURCES})
    target_link_libraries(${NAME} PRIVATE orange::core SDL3::SDL3)
    target_compile_features(${NAME} PRIVATE cxx_std_17)

    # Plugins are loaded at runtime (no link dependency), but they must exist in
    # the shared output dir when the app runs, so build them alongside it.
    foreach(_plugin render_gl render_vk)
        if(TARGET ${_plugin})
            add_dependencies(${NAME} ${_plugin})
        endif()
    endforeach()

    # On Windows, SDL3 is a DLL -- copy it next to the exe so it runs from the
    # build dir without a PATH dance.
    if(WIN32)
        add_custom_command(TARGET ${NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:SDL3::SDL3> $<TARGET_FILE_DIR:${NAME}>
            VERBATIM)
    endif()
endfunction()
