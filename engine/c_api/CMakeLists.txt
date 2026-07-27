# orange_c -- the stable C ABI wrapper around orange_core.
#
# Always SHARED: its whole reason to exist is being loadable from a different
# toolchain (or a different language) than the one that built the engine, which
# only holds across a DLL/.so boundary with a C interface. orange_core is linked
# PRIVATE so none of its C++/Eigen/SDL interface leaks to consumers -- a
# consumer of orange_c needs exactly one header (orange/c/orange.h) and one
# import library.
add_library(orange_c SHARED src/orange_c.cpp)
add_library(orange::c ALIAS orange_c)

target_compile_definitions(orange_c
    PRIVATE
        ORANGE_C_BUILD=1                      # -> __declspec(dllexport)
        ORANGE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        ORANGE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        ORANGE_VERSION_PATCH=${PROJECT_VERSION_PATCH})

target_include_directories(orange_c PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>")

target_link_libraries(orange_c PRIVATE orange::core)
target_compile_features(orange_c PRIVATE cxx_std_17)

set_target_properties(orange_c PROPERTIES
    OUTPUT_NAME               "orange_c"
    CXX_VISIBILITY_PRESET     hidden          # only the annotated C entry points
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    VERSION                   "${PROJECT_VERSION}"
    SOVERSION                 "${PROJECT_VERSION_MAJOR}")

# No DEBUG_POSTFIX: the file name is part of the ABI contract for consumers that
# LoadLibrary("orange_c.dll") by name. Debug and Release therefore need separate
# prefixes (which is what scripts/package_sdk.ps1 produces).
