# Running the tests — quick reference

Full details + coverage: [TESTING.md](TESTING.md). This is the cheat sheet.

## One-shot (configure + build + test)

```powershell
scripts/run_tests.ps1            # Windows (PowerShell)
```
```sh
scripts/run_tests.sh             # bash
```

## After a code change (build, then test)

```sh
cmake --build D:/Library/Orange/build --config Debug
ctest --test-dir D:/Library/Orange/build -C Debug --output-on-failure
```

> This Windows box needs the Visual Studio generator (no full MSVC env on PATH).
> First-time configure: `cmake -S D:/Library/Orange -B D:/Library/Orange/build -G "Visual Studio 16 2019" -A x64`

## Run a suite directly (per-check log)

```sh
D:/Library/Orange/build/bin/Debug/orange_tests.exe        # CPU toolkit
D:/Library/Orange/build/bin/Debug/orange_gui_tests.exe    # GUI behavior (headless)
```

Exit code `0` = all passed, `1` = a check failed.

## Suites

| CTest target       | What it checks |
|--------------------|----------------|
| `orange_tests`     | CPU toolkit: geometry, morton, color, sparse grid, modes, IO |
| `orange_gui_tests` | GUI behavior headless: selection toolbar, picking (single/box/filter/modifier/vertex), visibility |

GUI **rendering** (pixels), the font-measured menu bar, and the feel of
lasso/paint/cross-section/camera framing are not auto-run — see the manual
checklist in [TESTING.md](TESTING.md).

## Notes

- Toggle the whole suite off at configure: `-DORANGE_BUILD_TESTS=OFF`.
- Tests are headless + deterministic (no window, GPU, time, or RNG).
- Add a test: drop a `test_xxx()` into `engine/tests/test_orange.cpp` (or
  `test_gui.cpp`) and call it from `main()` — no CMake change needed.
