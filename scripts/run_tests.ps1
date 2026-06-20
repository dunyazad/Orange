# Configure, build, and run the Orange headless unit tests (CTest).
# Usage:  scripts/run_tests.ps1 [-Config Debug|Release]
param([string]$Config = "Debug")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

cmake -S $root -B "$root/build" -G "Visual Studio 16 2019" -A x64
cmake --build "$root/build" --config $Config --target orange_tests
ctest --test-dir "$root/build" -C $Config --output-on-failure
