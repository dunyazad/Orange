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

# Numeric build check on a real PLY (manual; needs an external file, not a CTest).
add_executable(orange_ply_check check_ply.cpp)
target_link_libraries(orange_ply_check PRIVATE orange::core)
set_target_properties(orange_ply_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
