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
