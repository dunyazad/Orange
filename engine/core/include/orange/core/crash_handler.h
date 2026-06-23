#pragma once

#include <string>

namespace orange::core {

// Install a process-wide crash handler. On an otherwise-fatal failure -- an
// access violation / segfault, or a std::terminate (unhandled C++ exception or
// abort/assert) -- it prints a symbolized stack trace to stderr AND to
// "<dir>/orange_crash.txt", so you can see exactly which function/file:line
// faulted instead of a silent exit. On Windows it also writes
// "<dir>/orange_crash.dmp", a minidump you can open in Visual Studio or WinDbg
// to inspect the full call stack and locals.
//
// File:line resolution needs debug symbols (PDBs); the build keeps them for both
// Debug and Release. `dir` empty => write next to the current working directory.
// Call once, as early as possible in main().
void installCrashHandler(const std::string& dir = "");

// --- Hang watchdog ---------------------------------------------------------
// A crash handler can't catch a *hang* (an infinite loop / deadlock freezes the
// main thread without raising an exception). The watchdog catches that: it runs a
// background thread that watches a heartbeat the main loop ticks every frame. If
// no heartbeat arrives for `timeoutSeconds`, it suspends the main thread, walks
// ITS stack, and writes "<dir>/orange_hang.txt" + "orange_hang.dmp" -- so you can
// see exactly where it is stuck -- then lets it keep running (it reports once per
// hang episode). Suppressed while a debugger is attached (pauses are expected).
//
// Usage: call installWatchdog() once (just before the run loop, after setup), then
// watchdogHeartbeat() every frame from the loop; stopWatchdog() on exit. Windows
// only; a no-op elsewhere.
void installWatchdog(double timeoutSeconds = 10.0, const std::string& dir = "");
void watchdogHeartbeat();
void stopWatchdog();

} // namespace orange::core
