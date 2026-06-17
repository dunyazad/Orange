#include "orange/core/console.h"

#if defined(_WIN32)

#include <windows.h>

#include <climits>
#include <cstdio>
#include <vector>

namespace orange::core {
namespace {

struct Monitor {
    RECT bounds;  // full monitor rect, in virtual-screen coordinates
    RECT work;    // work area (excludes taskbar)
};

BOOL CALLBACK collectMonitor(HMONITOR h, HDC, LPRECT, LPARAM data) {
    auto* mons = reinterpret_cast<std::vector<Monitor>*>(data);
    MONITORINFO mi;
    mi.cbSize = sizeof(mi);
    if (GetMonitorInfo(h, &mi)) mons->push_back({mi.rcMonitor, mi.rcWork});
    return TRUE;  // keep enumerating
}

long area(const RECT& r) {
    return static_cast<long>(r.right - r.left) * (r.bottom - r.top);
}

} // namespace

void setupConsoleWindow() {
    HWND con = GetConsoleWindow();
    if (!con) {
        // GUI launch (or detached): make our own console and wire up stdio.
        if (!AllocConsole()) return;
        FILE* f = nullptr;
        freopen_s(&f, "CONOUT$", "w", stdout);
        freopen_s(&f, "CONOUT$", "w", stderr);
        freopen_s(&f, "CONIN$",  "r", stdin);
        con = GetConsoleWindow();
        if (!con) return;
    }

    std::vector<Monitor> mons;
    EnumDisplayMonitors(nullptr, nullptr, collectMonitor,
                        reinterpret_cast<LPARAM>(&mons));
    if (mons.empty()) return;

    // Largest monitor by pixel area.
    size_t largest = 0;
    long bestArea = -1;
    for (size_t i = 0; i < mons.size(); ++i) {
        if (area(mons[i].bounds) > bestArea) {
            bestArea = area(mons[i].bounds);
            largest = i;
        }
    }

    // Monitor immediately to the right of the largest: the one whose left edge
    // is closest while still at/past the largest monitor's right edge. If there
    // is none, fall back to the largest monitor itself.
    const long rightEdge = mons[largest].bounds.right;
    size_t target = largest;
    long bestLeft = LONG_MAX;
    for (size_t i = 0; i < mons.size(); ++i) {
        if (i == largest) continue;
        const long left = mons[i].bounds.left;
        if (left >= rightEdge && left < bestLeft) {
            bestLeft = left;
            target = i;
        }
    }

    const RECT& work = mons[target].work;
    const int workW = work.right - work.left;
    const int workH = work.bottom - work.top;

    // A maximized console grows only as wide as its screen buffer has columns, so
    // widen the buffer to span the work area; otherwise the window is clamped far
    // short of the monitor. Grow only (a buffer may exceed its window), so there
    // is no need to resize the window first.
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO sbi;
    if (hOut != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(hOut, &sbi)) {
        COORD cell = {8, 16};  // fallback cell size if the font query fails
        CONSOLE_FONT_INFO cfi;
        if (GetCurrentConsoleFont(hOut, FALSE, &cfi)) {
            COORD c = GetConsoleFontSize(hOut, cfi.nFont);
            if (c.X > 0 && c.Y > 0) cell = c;
        }
        SHORT cols = static_cast<SHORT>(workW / cell.X);
        if (cols > sbi.dwSize.X) {
            COORD buf = {cols, sbi.dwSize.Y};  // keep the existing scrollback
            SetConsoleScreenBufferSize(hOut, buf);
        }
    }

    // Move the console onto the target monitor, then maximize it there. Maximize
    // snaps to whichever monitor the window currently sits on, so move first. The
    // widened buffer lets the maximized window fill the full monitor width.
    ShowWindow(con, SW_RESTORE);
    SetWindowPos(con, nullptr, work.left, work.top, workW, workH,
                 SWP_NOZORDER | SWP_NOACTIVATE);
    ShowWindow(con, SW_MAXIMIZE);
}

} // namespace orange::core

#else  // non-Windows: no console management

namespace orange::core {
void setupConsoleWindow() {}
} // namespace orange::core

#endif
