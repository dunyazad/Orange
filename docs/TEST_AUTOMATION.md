# Test & evaluation automation — the scheme

Goal: stop eyeballing screenshots. Every feature gets an **automatic check** with a
**pass/fail or a measured metric**, run from one command. Features split by how
they can be observed; each layer has a harness.

```
        what it checks                         harness / target            verdict
  ----  -------------------------------------  --------------------------  -----------------
  L1    pure CPU logic (math, geometry, IO)    orange_tests                assert pass/fail
  L2    GUI behavior (input -> ECS state)      orange_gui_tests            assert pass/fail
  L3    data-structure *properties*            orange_spatial_tests        assert pass/fail
  L4    performance + quality metrics          orange_bench                metric vs threshold
  L5    rendered pixels (visual features)      orange_shots + image diff   golden-image diff
```

L1-L4 are headless (no window/GPU) and already wired to CTest. L5 needs a GPU but
is scripted (no human) — see §5.

---

## 1. L1 — CPU logic (`orange_tests`)
Geometry, Morton, color, sparse grid, modes, IO. Brute-force or known-answer
assertions. (See TESTING.md.)

## 2. L2 — GUI behavior (`orange_gui_tests`)
The selection toolbar and picking are ECS input systems taking
`(registry, Input, w, h)` — no renderer. Feed a synthetic `Input`, assert the
resulting `Renderable.selected` / `ElementSelection` / `SelectionMode`. Covers:
single/box/lasso/paint, filter, modifier, vertex/edge/face, Ctrl/Shift/Alt,
visibility (Ctrl+A). Menu *dispatch* is covered by asserting `MenuBar.triggered`.

## 3. L3 — data-structure properties (`orange_spatial_tests`)
Every accelerator (BVH, Octree, KD, Uniform grid, Loose octree, BSP, R-tree, Ball
tree) checked against brute force for **correctness** (same query results). Plus
**structural invariants** that catch the bugs we hit by eye:

- **coverage** — every input point lands in some leaf/cell (catches "not all drawn").
- **outlier robustness** — with a few far outliers added, the *inlier* region's
  cell sizes stay small (catches the AABB-inflation that fattened lines / left the
  model under-subdivided).
- **box count** — a dense cloud yields > N nodes at the visualization depth
  (catches "octree only draws part").

These are deterministic and headless — the visualization bugs become unit tests
instead of screenshots.

## 4. L4 — performance & quality (`orange_bench`)
Not pass/fail-only: measures and prints a table, and asserts a few regression
guards (build completes, query is faster than brute, correctness holds). Metrics
per structure on synthetic clouds of growing size:

- build time (ms), bytes/structure estimate
- radius/kNN query time vs brute-force → **speedup ×**
- correctness (result set equals brute)

Run: `./build/bin/Debug/orange_bench` (or via CTest as `orange_bench`). Use the
printed speedups to evaluate which structure fits a workload; the guards fail CI
if a change makes a structure slower than brute or incorrect.

## 5. L5 — rendered pixels (golden-image), proposed
Visual features (grid scaling/fade, color modes, lighting, cross-section, the
selection silhouette, element-selection markers, the spatial overlays, the menu /
toolbar) can't be judged from ECS state — they live in pixels. Scheme:

1. **Deterministic scenes.** Add an `appOrange --selftest <dir>` mode: it builds a
   fixed set of canned scenes (load a bundled test mesh, set camera to a fixed
   pose, toggle each feature: grid on/off, each color mode, cross-section, each
   draw mode, each Spatial kind, a scripted box-selection), renders each, and
   writes `shot_<name>.png` via `IRenderer::readPixels`. No interaction, fixed
   seed, fixed viewport → reproducible images.
2. **Golden store.** Commit blessed references under `tests/golden/`.
3. **Diff.** A tiny `orange_imgdiff a.png b.png` (load via stb_image) computes the
   fraction of pixels differing beyond a per-channel tolerance; fail if it exceeds
   a threshold (e.g. 0.5%). Anti-aliasing/driver noise handled by the tolerance.
4. **Update flow.** `--selftest` writes to a scratch dir; a `--bless` flag copies
   scratch → golden after a human approves once.

This makes "lines too thick", "octree not fully drawn", "marker too big" into a
red diff against the golden, caught in CI instead of by eye. GPU/driver variance
means L5 runs on a fixed reference machine (or with generous tolerance), so L1-L4
remain the hard gate and L5 is an advisory visual gate.

### Status
- L1-L4: implemented and wired to CTest.
- L5: design above; `--selftest` + `orange_imgdiff` are the next build step (not yet
  implemented). Until then, visual features use the manual checklist in TESTING.md.

---

## Running everything

```sh
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure   # L1-L4 gate
./build/bin/Debug/orange_bench                          # L4 metric table
```

## Per-feature coverage matrix

| Feature                          | Layer | Target |
|----------------------------------|-------|--------|
| Geometry/Morton/color/IO         | L1    | orange_tests |
| Processing modes run             | L1    | orange_tests |
| Picking single/region/element    | L2    | orange_gui_tests |
| Selection filter/modifier/keys   | L2    | orange_gui_tests |
| Toolbar mode switching           | L2    | orange_gui_tests |
| Menu action dispatch             | L2    | orange_gui_tests |
| BVH/Octree/KD/Grid/Loose/BSP/RTree/Ball correctness | L3 | orange_spatial_tests |
| Structure coverage / robustness  | L3    | orange_spatial_tests |
| Structure build/query perf       | L4    | orange_bench |
| Camera fit / grid / cross-section / color / lighting / spatial overlay (pixels) | L5 | orange_shots + orange_imgdiff (proposed) |
