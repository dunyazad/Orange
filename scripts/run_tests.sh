#!/usr/bin/env bash
# Configure, build, and run the Orange headless unit tests (CTest).
# Usage:  scripts/run_tests.sh [Debug|Release]
set -euo pipefail

config="${1:-Debug}"
root="$(cd "$(dirname "$0")/.." && pwd)"

# Use the platform default generator; on the Windows dev box pass
# -G "Visual Studio 16 2019" -A x64 (see CLAUDE.md / run_tests.ps1).
cmake -S "$root" -B "$root/build"
cmake --build "$root/build" --config "$config" --target orange_tests
ctest --test-dir "$root/build" -C "$config" --output-on-failure
