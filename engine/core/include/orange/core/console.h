#pragma once

namespace orange::core {

// Ensures a console window exists (allocating one if the process has none) and
// maximizes it on the monitor immediately to the right of the largest monitor.
// No-op on non-Windows platforms.
void setupConsoleWindow();

} // namespace orange::core
