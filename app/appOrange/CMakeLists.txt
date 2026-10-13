# appOrange: a demo application built on the Orange engine. orange_add_app()
# (from engine/cmake/OrangeApp.cmake) links orange::core + SDL3, makes the
# render plugins build alongside it, and copies SDL3.dll next to the exe.
orange_add_app(appOrange SOURCES src/main.cpp)

# App-specific: this demo bakes its font atlas with stb_truetype.
target_include_directories(appOrange PRIVATE ${STB_INCLUDE_DIR})

# Declarative UI files (ui/*.json) are read at runtime from <exe dir>/ui/, so
# copy them next to the exe after every build (edits need no reconfigure).
add_custom_command(TARGET appOrange POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/ui" "$<TARGET_FILE_DIR:appOrange>/ui"
    VERBATIM)
