#include "orange/core/crash_handler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>

#if defined(_WIN32)

// clang-format off
#include <windows.h>
#include <dbghelp.h>
// clang-format on
#pragma comment(lib, "dbghelp.lib")

namespace orange::core {
namespace {

std::string g_dir;

std::string crashPath(const char* name) {
    return g_dir.empty() ? std::string(name) : g_dir + "\\" + name;
}

// Walk + symbolize the stack of `thread` (whose CONTEXT is `ctx`), writing each
// frame (function, and file:line when a PDB is available) to stderr and `f`. For
// a crash the thread is the current one; for the watchdog it is the suspended
// main thread.
void writeTrace(HANDLE thread, CONTEXT* ctx, std::FILE* f) {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(proc, nullptr, TRUE);

    STACKFRAME64 frame = {};
    DWORD machine = 0;
#if defined(_M_X64)
    machine = IMAGE_FILE_MACHINE_AMD64;
    frame.AddrPC.Offset    = ctx->Rip;
    frame.AddrFrame.Offset = ctx->Rbp;
    frame.AddrStack.Offset = ctx->Rsp;
#elif defined(_M_ARM64)
    machine = IMAGE_FILE_MACHINE_ARM64;
    frame.AddrPC.Offset    = ctx->Pc;
    frame.AddrFrame.Offset = ctx->Fp;
    frame.AddrStack.Offset = ctx->Sp;
#elif defined(_M_IX86)
    machine = IMAGE_FILE_MACHINE_I386;
    frame.AddrPC.Offset    = ctx->Eip;
    frame.AddrFrame.Offset = ctx->Ebp;
    frame.AddrStack.Offset = ctx->Esp;
#endif
    frame.AddrPC.Mode    = AddrModeFlat;
    frame.AddrFrame.Mode = AddrModeFlat;
    frame.AddrStack.Mode = AddrModeFlat;

    char symBuf[sizeof(SYMBOL_INFO) + 512] = {};
    auto* sym         = reinterpret_cast<SYMBOL_INFO*>(symBuf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen   = 512;

    auto out = [&](const char* s) { std::fputs(s, stderr); if (f) std::fputs(s, f); };
    out("\n=== Orange crash: stack trace (most recent call first) ===\n");
    for (int i = 0; i < 64; ++i) {
        if (!StackWalk64(machine, proc, thread, &frame, ctx, nullptr,
                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr))
            break;
        if (frame.AddrPC.Offset == 0) break;

        char buf[1200];
        DWORD64 disp = 0;
        if (SymFromAddr(proc, frame.AddrPC.Offset, &disp, sym)) {
            IMAGEHLP_LINE64 line = {};
            line.SizeOfStruct    = sizeof(line);
            DWORD lineDisp       = 0;
            if (SymGetLineFromAddr64(proc, frame.AddrPC.Offset, &lineDisp, &line))
                std::snprintf(buf, sizeof(buf), "  #%-2d %s  (%s:%lu)\n", i, sym->Name,
                              line.FileName, line.LineNumber);
            else
                std::snprintf(buf, sizeof(buf), "  #%-2d %s + 0x%llx\n", i, sym->Name,
                              (unsigned long long)disp);
        } else {
            std::snprintf(buf, sizeof(buf), "  #%-2d 0x%llx (no symbol)\n", i,
                          (unsigned long long)frame.AddrPC.Offset);
        }
        out(buf);
    }
    out("==========================================================\n");
    out("(a matching .txt + .dmp were written -- open the .dmp in Visual Studio / WinDbg)\n");
    SymCleanup(proc);
}

void writeMiniDump(EXCEPTION_POINTERS* ep, const char* name) {
    HANDLE file = CreateFileA(crashPath(name).c_str(), GENERIC_WRITE, 0, nullptr,
                              CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    MINIDUMP_EXCEPTION_INFORMATION mei = {};
    mei.ThreadId          = GetCurrentThreadId();
    mei.ExceptionPointers = ep;
    mei.ClientPointers    = FALSE;
    MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), file,
                      MiniDumpWithIndirectlyReferencedMemory, ep ? &mei : nullptr, nullptr,
                      nullptr);
    CloseHandle(file);
}

LONG WINAPI exceptionFilter(EXCEPTION_POINTERS* ep) {
    std::FILE* f = nullptr;
    fopen_s(&f, crashPath("orange_crash.txt").c_str(), "w");
    char hdr[160];
    std::snprintf(hdr, sizeof(hdr), "Orange: unhandled exception 0x%08lx at 0x%llx\n",
                  ep->ExceptionRecord->ExceptionCode,
                  (unsigned long long)ep->ExceptionRecord->ExceptionAddress);
    std::fputs(hdr, stderr);
    if (f) std::fputs(hdr, f);
    writeTrace(GetCurrentThread(), ep->ContextRecord, f);
    if (f) std::fclose(f);
    writeMiniDump(ep, "orange_crash.dmp");
    return EXCEPTION_EXECUTE_HANDLER;  // log, then let the process die
}

void terminateHandler() {
    std::FILE* f = nullptr;
    fopen_s(&f, crashPath("orange_crash.txt").c_str(), "w");
    const char* msg = "Orange: std::terminate (unhandled C++ exception / abort / assert)\n";
    std::fputs(msg, stderr);
    if (f) std::fputs(msg, f);
    CONTEXT ctx = {};
    RtlCaptureContext(&ctx);
    writeTrace(GetCurrentThread(), &ctx, f);
    if (f) std::fclose(f);
    std::abort();
}

// --- Hang watchdog ---------------------------------------------------------
std::thread          g_watchThread;
std::atomic<bool>    g_watchRun{false};
std::atomic<uint64_t> g_beat{0};       // bumped by watchdogHeartbeat() each frame
HANDLE               g_mainThread = nullptr;
double               g_timeout    = 10.0;

// Suspend the (hung) main thread, dump its stack + a minidump, resume it.
void captureHang() {
    std::FILE* f = nullptr;
    fopen_s(&f, crashPath("orange_hang.txt").c_str(), "w");
    const char* msg = "Orange: main thread appears HUNG (no heartbeat). Where it is stuck:\n";
    std::fputs(msg, stderr);
    if (f) std::fputs(msg, f);

    if (g_mainThread && SuspendThread(g_mainThread) != (DWORD)-1) {
        CONTEXT ctx;
        std::memset(&ctx, 0, sizeof(ctx));
        ctx.ContextFlags = CONTEXT_FULL;
        if (GetThreadContext(g_mainThread, &ctx)) writeTrace(g_mainThread, &ctx, f);
        ResumeThread(g_mainThread);
    }
    if (f) std::fclose(f);
    writeMiniDump(nullptr, "orange_hang.dmp");  // captures all threads
}

void watchLoop() {
    uint64_t lastSeen = g_beat.load();
    auto     lastChange = std::chrono::steady_clock::now();
    bool     reported = false;
    while (g_watchRun.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        uint64_t cur = g_beat.load();
        auto     now = std::chrono::steady_clock::now();
        if (cur != lastSeen) { lastSeen = cur; lastChange = now; reported = false; continue; }
        if (IsDebuggerPresent()) { lastChange = now; continue; }  // breakpoints aren't hangs
        double idle = std::chrono::duration<double>(now - lastChange).count();
        if (!reported && idle > g_timeout) {
            captureHang();
            reported = true;  // one dump per hang episode (reset when beats resume)
        }
    }
}

} // namespace

void installCrashHandler(const std::string& dir) {
    g_dir = dir;
    SetUnhandledExceptionFilter(exceptionFilter);
    std::set_terminate(terminateHandler);
}

void installWatchdog(double timeoutSeconds, const std::string& dir) {
    if (timeoutSeconds <= 0.0 || g_watchRun.load()) return;
    if (!dir.empty()) g_dir = dir;
    g_timeout = timeoutSeconds;
    // A real, watch-thread-usable handle to the current (main) thread.
    DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
                    &g_mainThread, 0, FALSE, DUPLICATE_SAME_ACCESS);
    g_watchRun.store(true);
    g_watchThread = std::thread(watchLoop);
}

void watchdogHeartbeat() { g_beat.fetch_add(1, std::memory_order_relaxed); }

void stopWatchdog() {
    if (!g_watchRun.exchange(false)) return;
    if (g_watchThread.joinable()) g_watchThread.join();
    if (g_mainThread) { CloseHandle(g_mainThread); g_mainThread = nullptr; }
}

} // namespace orange::core

#elif defined(__GNUC__) || defined(__clang__)

// POSIX best-effort: print a backtrace on fatal signals + std::terminate.
#include <csignal>
#include <cstring>
#include <execinfo.h>
#include <unistd.h>

namespace orange::core {
namespace {

std::string g_dir;

void dumpBacktrace(const char* reason) {
    std::FILE* f = std::fopen((g_dir.empty() ? std::string("orange_crash.txt")
                                             : g_dir + "/orange_crash.txt").c_str(), "w");
    auto out = [&](const char* s) { std::fputs(s, stderr); if (f) std::fputs(s, f); };
    out("\n=== Orange crash: ");
    out(reason);
    out(" ===\n");
    void* frames[64];
    int n = backtrace(frames, 64);
    char** syms = backtrace_symbols(frames, n);
    for (int i = 0; i < n; ++i) {
        if (syms) { out("  "); out(syms[i]); out("\n"); }
    }
    out("(run under a debugger, or build with -g, for file:line)\n");
    if (syms) std::free(syms);
    if (f) std::fclose(f);
}

void signalHandler(int sig) {
    dumpBacktrace(strsignal(sig));
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

void terminateHandler() {
    dumpBacktrace("std::terminate");
    std::abort();
}

} // namespace

void installCrashHandler(const std::string& dir) {
    g_dir = dir;
    std::signal(SIGSEGV, signalHandler);
    std::signal(SIGABRT, signalHandler);
    std::signal(SIGFPE, signalHandler);
    std::signal(SIGILL, signalHandler);
    std::set_terminate(terminateHandler);
}

// Watchdog stack-walk of an arbitrary thread isn't portable; no-op for now.
void installWatchdog(double, const std::string&) {}
void watchdogHeartbeat() {}
void stopWatchdog() {}

} // namespace orange::core

#else

namespace orange::core {
void installCrashHandler(const std::string&) {}
void installWatchdog(double, const std::string&) {}
void watchdogHeartbeat() {}
void stopWatchdog() {}
}

#endif
