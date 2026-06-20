# Testing Orange

Orange splits into a **headless CPU toolkit** (`orange_core`) and a **GUI app**
(`appOrange`, GL/VK rendering + ECS UI). The toolkit is covered by automated unit
tests; the GUI — rendering, the menu bar, the selection toolbar, picking — is
covered by the manual checklist at the end (it needs a window, a GPU, and a
human eye, so it isn't auto-run).

---

## 1. Automated unit tests

Headless, deterministic, no external test framework (deps are network-fetched, so
the harness is a tiny in-file assert runner). Two CTest targets, both linking
`orange::core`:

- **`orange_tests`** — the CPU toolkit (geometry, color, modes, IO, ...).
- **`orange_gui_tests`** — the **GUI behavior**: the selection toolbar and picking
  driven headlessly. The ECS input systems take only `(registry, Input, w, h)` —
  no window or renderer — so the selection logic is fully automatable by feeding a
  synthetic `core::Input` and asserting the resulting component state.

### Run

PowerShell / bash helper (configure + build + test):

```sh
scripts/run_tests.ps1      # Windows (PowerShell)
scripts/run_tests.sh       # bash
```

Or by hand (this machine uses the Visual Studio generator — see CLAUDE.md):

```sh
cmake -S . -B build -G "Visual Studio 16 2019" -A x64
cmake --build build --config Debug --target orange_tests
ctest --test-dir build -C Debug --output-on-failure
```

Run the exe directly for the per-check log: `./build/bin/Debug/orange_tests.exe`
(exit code 0 = all passed, 1 = a check failed).

Turn the suite off with `-DORANGE_BUILD_TESTS=OFF` at configure time.

### What's covered

`engine/tests/test_orange.cpp` (CPU toolkit):

| Group         | Subject                          | Sample assertions |
|---------------|----------------------------------|-------------------|
| `geometry`    | `Ray`/`AABB` (`geometry.h`)      | ray–triangle hit `t`, behind-ray miss, ray–sphere, AABB contains / ray slab / overlap |
| `morton`      | `Morton3D` (`morton3d.h`)        | encode→decode round-trip, `positionToIndex` floor |
| `color`       | `color.h`                        | HSV→RGB (hue 0 = red, S=0 = gray), heatmap ends, `Lerp`, palette size, deterministic `RandomFromIndex` |
| `sparse_grid` | `SparseGrid` (`sparse_grid.h`)   | radius query count on a lattice, `closestPoint`, kNN sorted nearest-first |
| `modes`       | `orange::modes` (`modes.h`)      | `modeCount`/`modeName`, every `runMode` runs, geometry emitted |
| `io`          | `XYZFormat` (`serialization.h`)  | point-cloud write→read round-trip |

`engine/tests/test_spatial.cpp` (spatial acceleration, vs brute force):

| Group          | Subject                          | Sample assertions |
|----------------|----------------------------------|-------------------|
| `bvh`          | `BVH` (`bvh.h`)                  | 200 rays' nearest hit `t` matches the brute triangle scan |
| `octree`       | `Octree` (`octree.h`)            | radius / AABB query counts, kNN distances vs brute |
| `kdtree`       | `KDTree` (`kdtree.h`)            | nearest index, kNN distances, radius set vs brute |
| `uniform_grid` | `UniformGrid` (`uniform_grid.h`) | radius count vs brute, occupied cells |
| `loose_octree` | `LooseOctree` (`loose_octree.h`) | radius count vs brute |
| `bsp`          | `BSP` (`bsp.h`)                  | radius count vs brute |
| `rtree`        | `RTree` (`rtree.h`)             | radius + AABB counts vs brute |
| `ball_tree`    | `BallTree` (`ball_tree.h`)       | radius count, kNN distances vs brute |

`engine/tests/test_gui.cpp` (GUI behavior, headless):

| Group           | Subject                       | Sample assertions |
|-----------------|-------------------------------|-------------------|
| `toolbar`       | `selectionToolbarInputSystem` | clicking a button sets the matching Target/Action/Filter/Modifier; bar captures the mouse |
| `pick.object`   | `pickingSystem` (Single)      | click selects the front object; empty click clears |
| `pick.filter`   | filter eligibility            | Point filter excludes a mesh; Mesh filter includes it |
| `pick.modifier` | Replace / Add / Subtract      | add keeps, subtract removes |
| `pick.box`      | Box drag                      | down→drag→release selects the enclosed object |
| `pick.vertex`   | element selection             | Vertex target picks the nearest vertex into `ElementSelection` |
| `visibility`    | `entityVisibleOnScreen`       | on-screen true, far-offscreen false (backs Ctrl+A) |

### Adding a test

Add a `static void test_xxx()` to `engine/tests/test_orange.cpp` using `CHECK` /
`CHECK_NEAR`, then call it from `main()`. No CMake change needed. Keep tests
headless (no window/GPU) and deterministic (no time/RNG — the modes and palettes
are seeded by index, not `rand`).

---

## 2. Manual GUI test checklist

The **selection toolbar mode-switching and picking behavior are now auto-covered**
by `orange_gui_tests` (§1). The manual pass focuses on what needs a GPU/eye:
pixel rendering, the font-measured **menu bar** layout, and the exact feel of
lasso/paint drags, cross-section, and camera framing.

Build + run the app (GL default, or `--vulkan`):

```sh
cmake --build build --config Debug --target appOrange
./build/bin/Debug/appOrange.exe            # OpenGL
./build/bin/Debug/appOrange.exe --vulkan   # Vulkan
```

Press **C** at any point to save `orange_capture.png` next to the exe for a visual
record. Run the whole list on **both** backends — they must look identical.

### Load & framing
- [ ] **File ▸ Open…** loads a mesh/point cloud; it renders at its **original
      coordinates** (not recentered).
- [ ] Camera **frames the loaded model** automatically (fills the view, not tiny,
      not clipped).
- [ ] **R** resets the camera to that framing.
- [ ] Wheel **zoom** feels consistent regardless of model size; nothing z-clips
      when zoomed fully in or out.
- [ ] **Ctrl + left-click** on the model glides the camera to recenter on that point.

### Grid
- [ ] Ground grid scales to the model (≈10 cells across); lines on round coords.
- [ ] Zooming out, the grid keeps filling the view (no hard-cut disc).
- [ ] **Space** (or View ▸ Ground Grid) toggles it; menu tick reflects state.

### Menu bar
- [ ] Each title opens its dropdown; hovering another title switches to it.
- [ ] Check items (Grid, Lighting, VSync, Cross-Section, Color) show a tick that
      matches the live state.
- [ ] Every item performs the same action as its keyboard shortcut.
- [ ] **Render ▸ Color** Original / Height / Position / Grayscale switch scene
      coloring (Shift+\` cycles the same).
- [ ] **Render ▸ Lighting** (\`), **VSync**, **Cross-Section** toggle on/off.
- [ ] **Draw** menu sets the selected mesh's mode (None/Solid/Wireframe/
      Wireframe+Solid/Point); **Tab** cycles it.
- [ ] **Points** menu cycles the processing mode (also **M**).

### Selection toolbar (left)
- [ ] Four groups visible — **Target** (Obj/Vtx/Edg/Fac), **Action**
      (Sgl/Box/Las/Pnt), **Filter** (All/Msh/Pts), **Modifier** (Set/+/−);
      active button highlighted.
- [ ] **Single (Sgl)**: click selects the front-most object; empty click clears.
- [ ] **Box**: drag a rectangle (rubber-band shows) → objects inside selected.
- [ ] **Lasso (Las)**: freehand drag → enclosed objects selected.
- [ ] **Paint (Pnt)**: drag over objects → continuously added.
- [ ] **Filter** restricts to Mesh-only / Point-only / All.
- [ ] **Modifier** Set replaces, **+** adds, **−** subtracts.
- [ ] **Target Vtx/Edg/Fac**: element selection shows lime markers (vertex squares
      sized to the point sprites, edge lines, face triangles).

### Keyboard modifiers during selection
- [ ] **Shift + drag/click** adds to the selection.
- [ ] **Alt + drag/click** subtracts.
- [ ] **Ctrl + drag** (Box/Lasso/Paint) subtracts; **Ctrl + single-click** still
      recenters the camera (does not select).

### Other
- [ ] **Cross-Section**: enable, drag the slider — the cut plane sweeps the model's
      full extent on the chosen axis (X/Y/Z, flip).
- [ ] **+ / −** resize point sprites; vertex markers track the size.
- [ ] **Ctrl+A** selects all on-screen; **Delete** removes the selection;
      **H** unhides None-mode meshes.
- [ ] **Y/Z up-axis** button (gizmo corner) stands a Z-up model upright.
- [ ] Axis **gizmo** click snaps the camera to that view.
- [ ] **Esc** quits.

### Performance sanity
- [ ] Box/Lasso vertex selection on a large point cloud completes promptly (the
      projection is parallel + the merge is set-based, not O(n²)).
- [ ] FPS readout stays reasonable with a large cloud loaded.
