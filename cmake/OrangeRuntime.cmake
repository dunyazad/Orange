# ---------------------------------------------------------------------------
# Consumer-side helpers, installed with the binary SDK and included by
# OrangeConfig.cmake. Nothing here is used by the Orange build itself.
# ---------------------------------------------------------------------------

# orange_copy_runtime(<target> [DESTINATION <dir>])
#
# Copies the shared runtime the engine needs at load time -- the render plugins
# (render_gl / render_vk) plus SDL3 and any other DLL shipped in the SDK's bin/
# -- next to <target>'s executable after it builds. The plugin loader searches
# the directory of the running executable, so this is what makes a consumer app
# runnable straight out of its build tree.
function(orange_copy_runtime TARGET)
    cmake_parse_arguments(OCR "" "DESTINATION" "" ${ARGN})
    if(NOT TARGET ${TARGET})
        message(FATAL_ERROR "orange_copy_runtime: '${TARGET}' is not a target")
    endif()

    set(_dest "$<TARGET_FILE_DIR:${TARGET}>")
    if(OCR_DESTINATION)
        set(_dest "${OCR_DESTINATION}")
    endif()

    file(GLOB _runtime
        "${ORANGE_PLUGIN_DIR}/*.dll"
        "${ORANGE_PLUGIN_DIR}/*.so"
        "${ORANGE_PLUGIN_DIR}/*.dylib")

    if(NOT _runtime)
        message(WARNING "orange_copy_runtime: no runtime files in ${ORANGE_PLUGIN_DIR}")
        return()
    endif()

    add_custom_command(TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_runtime} "${_dest}"
        COMMENT "Copying Orange render plugins + SDL3 next to ${TARGET}"
        VERBATIM)
endfunction()

# orange_add_app(<name> SOURCES <src>...)
#
# The SDK counterpart of the in-tree orange_add_app(): an executable that links
# the prebuilt orange::core and gets the render plugins copied beside it. Use it
# for a full Orange application; for a library or a tool that only wants the
# CPU geometry/IO toolkit, just target_link_libraries(... orange::core).
function(orange_add_app NAME)
    cmake_parse_arguments(APP "" "" "SOURCES" ${ARGN})
    if(NOT APP_SOURCES)
        message(FATAL_ERROR "orange_add_app(${NAME}): no SOURCES given")
    endif()

    add_executable(${NAME} ${APP_SOURCES})
    target_link_libraries(${NAME} PRIVATE orange::core)
    target_compile_features(${NAME} PRIVATE cxx_std_17)
    orange_copy_runtime(${NAME})
endfunction()
