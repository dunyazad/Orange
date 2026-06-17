# The sandbox application: links the core engine, loads a render plugin at run.
add_executable(sandbox src/main.cpp)

target_link_libraries(sandbox
    PRIVATE
        orange::core
        SDL3::SDL3)

target_include_directories(sandbox PRIVATE ${STB_INCLUDE_DIR})  # stb_truetype

target_compile_features(sandbox PRIVATE cxx_std_17)

# Make sure the plugins are built before (and alongside) the app so they sit in
# the same output directory the loader searches.
if(TARGET render_gl)
    add_dependencies(sandbox render_gl)
endif()
if(TARGET render_vk)
    add_dependencies(sandbox render_vk)
endif()

# On Windows, copy SDL3.dll next to the executable so it runs from the build dir.
if(WIN32)
    add_custom_command(TARGET sandbox POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL3::SDL3> $<TARGET_FILE_DIR:sandbox>
        VERBATIM)
endif()
