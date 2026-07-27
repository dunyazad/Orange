# Headless unit tests for orange_core (CPU toolkit). Registered with CTest.
# Built when ORANGE_BUILD_TESTS is ON (default). Run with: ctest --test-dir build
add_executable(orange_tests test_orange.cpp)
target_link_libraries(orange_tests PRIVATE orange::core)
set_target_properties(orange_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME orange_tests COMMAND orange_tests)

# Headless GUI behavior tests: drive the ECS input systems (selection toolbar,
# picking) with a synthetic Input -- no window/renderer needed.
add_executable(orange_gui_tests test_gui.cpp)
target_link_libraries(orange_gui_tests PRIVATE orange::core EnTT::EnTT)
set_target_properties(orange_gui_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME orange_gui_tests COMMAND orange_gui_tests)

# Spatial acceleration structures (BVH / Octree / KD-tree) vs brute force.
add_executable(orange_spatial_tests test_spatial.cpp)
target_link_libraries(orange_spatial_tests PRIVATE orange::core)
set_target_properties(orange_spatial_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME orange_spatial_tests COMMAND orange_spatial_tests)

# L4 evaluation harness: build/query timings + speedup vs brute (also a guard).
add_executable(orange_bench bench_spatial.cpp)
target_link_libraries(orange_bench PRIVATE orange::core)
set_target_properties(orange_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME orange_bench COMMAND orange_bench)

# Poisson reconstruction + pick-BVH dangling regression guard (cached BVH must
# survive entity destruction relocating the PickGeometry/PickBVH pools).
add_executable(orange_poisson_tests poisson_tests.cpp)
target_link_libraries(orange_poisson_tests PRIVATE orange::core EnTT::EnTT)
set_target_properties(orange_poisson_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(NAME orange_poisson_tests COMMAND orange_poisson_tests)

# Headless mode-runner on a real PLY: runs any Recolor mode, prints class
# counts, writes a recolored PLY for the --shot viewer (manual, not a CTest).
add_executable(orange_margin_probe probe_margin.cpp)
target_link_libraries(orange_margin_probe PRIVATE orange::core)
set_target_properties(orange_margin_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# C ABI wrapper (orange/c/orange.h), compiled as C against the wrapper source
# with ORANGE_C_STATIC -- same entry points orange_c.dll exports, but linked into
# the test exe so the suite does not depend on loading the DLL.
if(ORANGE_BUILD_C_API)
    add_executable(orange_c_tests test_c_api.c
                   "${CMAKE_CURRENT_SOURCE_DIR}/../c_api/src/orange_c.cpp")
    target_compile_definitions(orange_c_tests PRIVATE
        ORANGE_C_STATIC=1
        ORANGE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        ORANGE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        ORANGE_VERSION_PATCH=${PROJECT_VERSION_PATCH})
    target_include_directories(orange_c_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../c_api/include")
    target_link_libraries(orange_c_tests PRIVATE orange::core)
    set_target_properties(orange_c_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME orange_c_tests COMMAND orange_c_tests)
endif()

# Numeric build check on a real PLY (manual; needs an external file, not a CTest).
add_executable(orange_ply_check check_ply.cpp)
target_link_libraries(orange_ply_check PRIVATE orange::core)
set_target_properties(orange_ply_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
