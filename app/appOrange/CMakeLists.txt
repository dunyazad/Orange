# appOrange: a demo application built on the Orange engine. orange_add_app()
# (from engine/cmake/OrangeApp.cmake) links orange::core + SDL3, makes the
# render plugins build alongside it, and copies SDL3.dll next to the exe.
orange_add_app(appOrange SOURCES src/main.cpp)

# App-specific: this demo bakes its font atlas with stb_truetype.
target_include_directories(appOrange PRIVATE ${STB_INCLUDE_DIR})
