# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

**Orange** — a C++17, ECS-driven application with a **plugin-based rendering
engine**. Render backends (OpenGL, Vulkan) are separate shared libraries
(`render_gl.dll`, `render_vk.dll`) loaded at runtime through a stable C ABI. The
app core never links GL/VK/SDL directly — it talks only to the abstract
`orange::render::IRenderer` (`engine/render_api`).

- **ECS, not a scene graph** — entities + components via [EnTT]. Systems
  transform the data; `renderSystem` is the bridge from ECS to draw calls.
- **Plugin boundary** — a backend exports three C symbols (`orangeRenderPluginInfo`,
  `orangeRenderCreate`, `orangeRenderDestroy`) and is checked against
  `ORANGE_PLUGIN_ABI_VERSION` at load.
- **Cross-platform by design** — [SDL3] handles window/input/GL-context/Vulkan-
  surface, so macOS/iOS/Android are additive, not a rewrite.

See `README.md` for the full architecture, the resource/buffer model, the axis
gizmo (ViewCube), and the controls.

## Geometry / IO / debug toolkit (ported from the Elements/Helium engine)

CPU, Eigen-backed, CUDA-free utilities that live in `orange_core` (headers under
`orange/core/`, sources under `engine/core/src/{geometry,io}/`):

- **`orange::geometry`** — `Morton3D` (Z-order keys), `Ray`/`AABB`
  (`geometry.h`), spatial acceleration structures — `BVH` (`bvh.h`, triangle
  ray pick, median-split AABB tree; the `pickingSystem` lazily builds + caches one
  per mesh in a `PickBVH` component so repeated picks are O(log n)), `Octree`
  (`octree.h`, point radius/AABB/kNN + region culling), `KDTree` (`kdtree.h`,
  point nearest/kNN/radius), `UniformGrid` (`uniform_grid.h`), `LooseOctree`
  (`loose_octree.h`), `BSP` (`bsp.h`, spatial-median), `RTree` (`rtree.h`, STR
  bulk-loaded), `BallTree` (`ball_tree.h`, bounding spheres). The **Spatial menu**
  visualizes the chosen structure on the *selected* mesh/point cloud: renderSystem
  builds it once into a static, per-level-colored wireframe mesh (cached in
  `SpatialVizCache`; ctx `SpatialViz` holds the kind) and draws it each frame —
  box structures as wire boxes, `BallTree` as spheres. — `SparseGrid` (hash grid; kNN + radius queries),
  `SparseDataBlock` (8³-block TSDF fused from an oriented point cloud via
  `fromPointsData`), and `generateMesh()` / `pointsToMesh()` — a dual-contouring
  surface extractor (the **points-in → mesh-out** pipeline). `std::execution::par`
  parallel; `marching_cubes_tables.h` holds the classic Bourke tables.
- **`orange::io`** — `serialization.h`: multi-format mesh/point-cloud read+write
  (XYZ/OFF/STL/OBJ/PLY/PTS/ALP/CSV) via `HSerializable`. `reconstructMeshFromFile()`
  bridges a loaded point cloud into `pointsToMesh`. (appOrange's `mesh_io.h` is the
  separate, render-facing OBJ/STL/PLY *loader* that fills `render::Vertex`.)
- **`orange::color`** — `color.h`: named colors, HSV↔RGB, heatmap, maximin
  contrasting palettes.
- **`orange::debug`** — `debug_draw.h`: immediate-mode `DebugDraw` singleton
  (`addLine/addBox/addWireBox/addSphere/addTriangle`). The renderSystem uploads
  the accumulated triangles as one dynamic non-indexed mesh each frame, then
  clears. Since the renderer has no line primitive, lines/wire boxes are emitted
  as thin triangle tubes.
- **Third-party:** `robin_hood` (header-only hash map) vendored under
  `engine/third_party/` (used by `SparseGrid`).

## Layout

```
engine/                  the reusable engine (CMake: add_subdirectory(engine))
  render_api/            interface-only C-ABI contract (header library) — no GL/VK/SDL
  core/                  platform layer + ECS (components, systems, app loop, window)
  plugins/render_gl/     OpenGL 3.3 backend plugin
  plugins/render_vk/     Vulkan backend plugin (built only if a Vulkan SDK is found)
  cmake/OrangeApp.cmake  orange_add_app() — the entry point every app uses
app/appOrange/           demo app: three spinning cubes via ECS (also hosts font/UI impl)
```

**Engine / app split.** The top-level CMake is a thin superbuild: it fetches deps,
then `add_subdirectory(engine)` (core + render plugins) and one `add_subdirectory`
per app. The engine never depends on any app, so multiple apps can sit on one
engine. An app is a thin consumer — `orange_add_app(<name> SOURCES ...)` links
`orange::core` + SDL3, builds the render plugins alongside it, and (on Windows)
copies SDL3.dll next to the exe. Apps talk only to `orange::core`, never GL/VK/SDL.
To add an app: new `app/<name>/` with a one-line `orange_add_app()` + an
`add_subdirectory(app/<name>)` in the root.

## Build & run

Requires CMake ≥ 3.21 and a C++17 compiler. SDL3, EnTT, and stb are fetched via
FetchContent on first configure (needs network). All binaries land in the bin
output dir so the app finds plugins next to itself.

Generic (per README):
```sh
cmake -S . -B build
cmake --build build --config Debug
./build/bin/appOrange            # OpenGL (default)
./build/bin/appOrange --vulkan   # Vulkan
```

### ⚠️ This Windows machine (D:\Library\Orange)

The shell has `cl`/`ninja` on PATH but **NOT** the full MSVC env (no
`INCLUDE`/`VCINSTALLDIR`), so **Ninja fails**. Use the Visual Studio generator:

```sh
cmake -S D:/Library/Orange -B D:/Library/Orange/build -G "Visual Studio 16 2019" -A x64
cmake --build D:/Library/Orange/build --config Debug
```

- Vulkan SDK is at `C:\VulkanSDK\1.4.350.0` (installed via
  `winget install KhronosGroup.VulkanSDK`), so `render_vk` builds here.
  `VULKAN_SDK` is set machine-wide, but a fresh shell may not have it yet —
  if CMake reports "Vulkan SDK not found", set `$env:VULKAN_SDK` for the
  session, delete `build/CMakeCache.txt`, and reconfigure.
- Binaries land in `build/bin/Debug`. Run `appOrange.exe` (GL) or
  `appOrange.exe --vulkan`.

## Key facts to keep in mind

- **Plugin ABI is v18.** v18 changed `IRenderer::drawGrid` to
  `drawGrid(int upAxis, float cellSize, const float cameraPos[3], float viewRadius)`:
  the grid now scales to the loaded model instead of a fixed ~3-unit assumption.
  `cellSize` is the world size of one minor cell (major lines every 10); the host
  derives it from the scene's world AABB (largest x/z extent → ~10 cells across,
  snapped to a power of ten). `cameraPos` + `viewRadius` move the distance fade from
  a fixed disc around the origin to a disc around the **camera** (radius ≈ orbit
  distance × 4), so the grid keeps filling the view at any zoom instead of cutting
  off when zoomed out. GL drives `uGridScale`/`uCamPos`/`uViewRadius` uniforms (needed
  a `glUniform3fv` entry in the hand-rolled `gl_loader`), VK fragment push constants
  at offsets 132 (cellSize) / 144 (camPos vec3) / 156 (viewRadius); the grid push
  range grew to 160 bytes. The host side lives in `systems.cpp` renderSystem, which
  computes the scene world AABB once and reuses it for the grid cell size AND the
  cross-section slider range. **Model framing (appOrange):** meshes load at their
  **original coordinates** (the old recenter-on-origin + fit-to-3-units step is gone);
  `finalizeMesh` instead frames the camera on the new mesh (target → bounds center,
  distance → fit the bounding sphere to the FOV, home pose updated so R reframes) and
  widens min/maxDistance. The camera's `zNear`/`zFar` are recomputed **every frame**
  from the orbit distance (`zNear = max(0.001, dist*0.005)`, `zFar = dist*50`) so
  nothing z-clips at any zoom. Wheel zoom is multiplicative
  (`distance *= pow(0.9, wheel*zoomSpeed)`) so it feels the same at any model scale;
  pan was already `panSpeed*distance`. The processing modes were already
  diag-relative. The cross-section slider range (`CrossSection::minPos/maxPos`) is
  set each frame from the scene AABB on the active axis (was hardcoded ±3).
- v17 added `IRenderer::setColorMode(uint32_t)` (Shift+`` ` ``
  cycles it): scene meshes/point clouds recolor in the fragment shader from the
  world position — 0 = original (vertex color), 1 = height (world Y → jet heatmap),
  2 = position (world XYZ → RGB), 3 = grayscale (luminance). **appOrange starts in
  grayscale (mode 3)** — `Application::colorMode_` defaults to 3 and is pushed at
  `run()` start. The grid/overlays always use mode 0 (GL zeroes `uColorMode` in
  `beginOverlay`; VK pushes 0 while `inOverlay_`). GL drives a `uColorMode` int
  uniform, VK a fragment push constant at offset 176. `setLighting` now shades
  triangle meshes too (flat normal from `dFdx/dFdy(worldPos)`), not just point
  sprites; overlays stay unshaded (GL zeroes `uLighting` in `beginOverlay`, VK pushes
  0 while `inOverlay_`). v16 added `IRenderer::setCrossSection(bool, const float plane[4])`:
  a world-space clipping plane `(nx,ny,nz,d)` — scene-mesh fragments where
  `dot(worldPos,n)+d > 0` are discarded, revealing a cut interior (applies to
  triangle meshes AND point clouds; the grid/overlays are never clipped). Both
  backends pass world position from the vertex stage to the fragment stage and
  discard there: GL via a `uClipPlane` vec4 uniform (zeroed in `beginOverlay` so
  overlays never clip), VK via a fragment push constant at offset 160 (pushed
  per-draw as zero while `inOverlay_`). The host drives it from a `CrossSection`
  component + slider panel (`crossSectionInputSystem`); `renderSystem` turns
  (enabled, axis, flip, pos) into the plane and calls `setCrossSection` before the
  scene submits. v13 added `MeshDesc::topology` (`Triangles` or `Points`);
  a `Points` mesh is drawn as **sphere-imposter point sprites** — real GL_POINTS /
  VK POINT_LIST with a fragment shader that rounds + shades each sprite via
  `gl_PointCoord`, not instanced spheres. Faceless PLYs load as such point clouds;
  their selection shows a bounding-box wireframe, not the stencil silhouette. v14
  added `IRenderer::setPointSize(float)` (the `+`/`-` keys resize point sprites;
  GL drives `gl_PointSize` via a `uPointSize` uniform, VK via a vertex push
  constant at offset 144). v15 added `IRenderer::setLighting(bool)` (the `` ` ``
  backtick key toggles it): the point sprite is the only lit primitive, so this
  lights/unlits the sphere-imposter diffuse shading (off => flat vertex color);
  triangle meshes are unlit (no normals) and the grid/overlays unaffected. GL
  drives a `uLighting` int uniform, VK a fragment push constant at offset 148.
  Changing the C contract means bumping
  `ORANGE_PLUGIN_ABI_VERSION` and updating both backends. v6 added
  `IRenderer::setVsync(bool)` (GL flips the swap interval; VK recreates the
  swapchain with a FIFO / immediate-mailbox present mode). v7 added
  `IRenderer::drawGrid()`. v8 made it `drawGrid(int upAxis)` (1 = Y up, 2 = Z up)
  — the axis arg only recolors the in-plane depth line, not the grid plane. v9
  added `IRenderer::setDrawMode(uint32_t)`. v10 redefined it to **Helium's
  Renderable drawing modes** (0 = none, 1 = solid, 2 = wireframe, 3 =
  wireframe-over-solid, 4 = point): GL mirrors Helium's `DrawImplementation`
  (`glPolygonMode` + `glPointSize` + `glPolygonOffset` for the over-solid pass);
  VK binds the matching fill/line/point pipeline (needs the `fillModeNonSolid` +
  `largePoints` device features; the fill pipeline carries a small depth bias so
  the over-solid edges sit on top, and mode 3 double-draws). v11 added
  `DrawItem::outline` and v12 `DrawItem::stencilMask` for the stencil selection
  silhouette (below).
- **Per-mesh drawing mode + selection.** Each `Renderable` carries its own
  `core::DrawMode` (so meshes keep distinct modes). The `renderSystem` calls
  `setDrawMode(r.drawMode)` per drawable; **Tab** cycles the mode of the *selected*
  meshes (`Application` iterates `Renderable::selected`). Selection shows as a
  **stencil silhouette outline** (a thin border that holds for any mesh shape and
  draw mode): for a selected mesh the renderSystem submits, after the visible mesh,
  a solid `DrawItem::stencilMask` pass (writes stencil = 1 over the footprint, no
  color/depth) then a slightly-enlarged `DrawItem::outline` pass drawn only where
  stencil != 1 (orange). GL drives `glStencil*` (needs `SDL_GL_STENCIL_SIZE`); VK
  uses a D24S8 depth-stencil attachment + stencil-write / stencil-test pipelines,
  clearing the stencil per mesh so multi-selection outlines don't clip. (The old
  inverted-hull version filled in on concave/thin meshes — stencil fixed that.)
  Picking: plain click single-selects (empty click clears), **Ctrl+left-click
  recenters** the camera on the picked point (`pickingSystem` computes the world
  hit `rayOW + rayDW*bestT` and starts a `CameraManipulator` target-glide;
  `cameraManipulatorSystem` eases `target` so the position follows — no selection
  change), **Ctrl+A** selects all meshes *visible on screen* (`entityVisibleOnScreen`
  frustum-tests each AABB, so off-screen meshes are never bulk-selected), **Delete**
  destroys the selected entities.
- **Infinite grid:** `renderSystem` calls `IRenderer::drawGrid(upAxis)` after the
  scene submits (before the gizmo overlay). Each backend renders a vertex-less
  full-screen pass (GL: inline shader + empty VAO; VK: a second pipeline +
  `shaders/grid.{vert,frag}` → SPIR-V) that ray-casts each pixel onto the world
  y=0 plane and draws `fwidth`-based AA grid lines with distance fade and correct
  depth (so the scene occludes it). `upAxis` colors the in-plane depth axis line
  blue (Z, Y-up) or green (Y, Z-up); GL passes a `uUpAxis` uniform, VK an `int` at
  push-constant offset 128. Each plugin inverts its own clip-corrected `viewProj`
  for the unprojection (the host doesn't, since VK's correction is plugin-side).
- **Up-axis (Y/Z) toggle:** a small "Y"/"Z" button in the gizmo's bottom-left
  corner (`AxisGizmo::zUp`, hit-tested in `axisGizmoInputSystem` via
  `upToggleRect`). It is a real coordinate-frame change, **not** a camera spin: a
  world up-axis basis `Mworld` (identity for Y up, `rotateX(-90°)` for Z up, so a
  model's logical +Z maps to render +Y) is prepended to every renderable's model
  in `renderSystem` and to the gizmo cube (`conjugate(camOrient)·Mworld`). The
  camera and the horizontal y=0 grid stay put; content is re-expressed in the new
  frame (a Z-up mesh stands up under Z up). The grid's depth-axis line recolors
  (blue Z → green Y) via the `drawGrid` arg. Picking undoes `Mworld` on the ray;
  gizmo cube-face snaps map the picked logical axis back through `Mworld`. Helpers
  `worldUpQuat/worldUpMatrix/worldZUp` live in `systems.cpp`.
- **Buffers are two-layer:** handle + byte-size based ABI in `render_api`, and a
  type-safe `core::Buffer<T>` (VertexBuffer/IndexBuffer/UniformBuffer) in
  `engine/core/include/orange/core/buffer.h` that wraps it with RAII. App code
  uses the template layer.
- **Vulkan shaders:** `engine/plugins/render_vk/shaders/*.{vert,frag}` are compiled to
  SPIR-V headers in `src/generated/` (CMake regenerates on change via the Vulkan
  SDK's glsl compiler). GL→VK clip-space correction (Y flip + depth 0..1) is
  applied so both backends show the identical scene.
- **Math is Eigen-backed.** Code uses the Eigen types directly —
  `Eigen::Vector3f/Vector4f/Matrix4f/Quaternionf` (column-major, GL convention);
  there are no `Vec3`/`Mat4`/... type aliases. `orange/core/math.h` keeps thin
  graphics helpers in `namespace orange::math` (`perspective`/`ortho`/`lookAt`/
  `quatAxisAngle`/`toMat4`/...). Eigen is fetched via FetchContent (find_package
  first). The build defines `EIGEN_MAX_ALIGN_BYTES=0` so Eigen members are safe
  inside EnTT pools. Interop with the render ABI's `float[16]`/`float[3]` is via
  `.data()`. Use `.x()/.y()/.z()` (methods), not `.x`, and construct with
  `Eigen::Vector3f(a,b,c)` / `Eigen::Quaternionf::Identity()` (the default Eigen
  ctor leaves values uninitialized).
- **No gimbal lock:** `Transform::orientation` is a unit quaternion; the camera
  is a quaternion arcball trackball (`CameraManipulator`). Any camera controller
  is just a system that writes a `Transform`.
- **Input scheme:** right-drag orbits, middle-drag pans, wheel zooms. **Left-click
  picks** the nearest `Renderable` via `pickingSystem` — a world ray is ray-tested
  against each entity's actual triangles when it has a `PickGeometry` component
  (CPU triangle soup), else its `Renderable` AABB; hits are compared by true world
  distance. Triangle picking means clicking through a concave model's gaps misses
  it and hits whatever is behind, instead of its bounding box stealing the click.
  Plain click single-selects (empty click clears), **Ctrl+left-click** recenters
  the camera on the picked point (animated target-glide, no selection change),
  **Ctrl+A** selects all on-screen meshes, **Delete**
  removes the selection. `Input::captured` (set by UI widgets) suppresses picking.
  **Tab** cycles the *selected* meshes' drawing mode
  (Helium's set: none / solid / wireframe / wireframe-over-solid / point);
  selection shows as a silhouette outline. **M** cycles the point-cloud processing
  mode; **`** (backtick) toggles point-sprite lighting (`IRenderer::setLighting`);
  **Shift+`** cycles the scene coloring mode (`IRenderer::setColorMode`:
  default/height/position/grayscale); **Space** toggles the ground grid
  (`GridState` in registry ctx, read by `renderSystem`); **R** resets the camera to
  its home pose (animated orientation + target + distance glide back to
  `CameraManipulator::home*`); **C** screenshots; **Esc** quits.
- **Processing modes / geometry operators (`orange::modes`):** the Orange take on
  Hydrogen's "apps" — selectable point-cloud operations that emit a visualization
  via the debug-draw accumulator. Each mode is tagged with a `ModeCategory`
  (**Generate / Analyze / Filter**); the **Geometry** menu is built from the
  registry, grouped by category with separators (so adding a mode = one function +
  one `kModes` row in `modes.cpp` + a contiguous `MenuAction::ModeN`, and it
  appears automatically). `processingModeSystem` (in `systems.cpp`) sources its
  input from the **first selected entity's** `PickGeometry` (vertex positions →
  world space), or from a ctx `modes::ModeInput` if one is present (tests/fixed
  clouds); it caches the result and recomputes only when the active mode
  (`modes::ModeState`) or the selection changes. **M** cycles the active mode. The
  ten ported, CUDA/Helium-free operators: *Generate* — **Reconstruct** (TSDF →
  `pointsToMesh`; estimates normals when absent), **SDF Filter** (UDF splat + box
  blur + iso-surface resample); *Analyze* (per-point scalar → heatmap) —
  **Clustering** (SparseGrid radius + union-find), **Curvature** (PCA surface
  variation λ0/Σλ), **Normal Deviation** (angle vs neighbourhood mean normal),
  **Density (KDE)** (Gaussian kernel); *Filter* (keep/drop, green/red) —
  **Outlier: SOR** (statistical), **Outlier: ROR** (radius count), **Outlier:
  PFOR** (plane-fit distance), **Morphology** (voxel erode + largest component);
  *Transform* — **Smooth (bilateral)** (edge-preserving Laplacian) and **ICP
  Register** (self-demo: perturb → realign, target blue / aligned green). The
  Analyze/Filter/normal ops share `orange::geometry::estimateNormals`
  (`normals.h`, SparseGrid kNN + PCA) and a kNN/PCA helper in `modes.cpp`; Smooth
  and ICP are CPU reimplementations of Helium's GPU-only versions, in
  `orange::geometry` (`point_ops.h`: `smoothPoints`, `icpAlign` — Eigen-SVD
  point-to-point). All self-scale to the input's bbox diagonal (fixed thresholds,
  no UI sliders yet).
- **Create menu (parametric primitives):** `orange::geometry::buildPlane/Box/
  Sphere/Cylinder/Cone/Torus/Disk/Capsule/Arrow` (`primitives.h`, the CPU port of
  Helium's GeometryBuilder) return a `geometry::Triangle` soup; `Application::
  applyMenuAction`'s `spawnPrimitive` uploads it to a GPU buffer/mesh and spawns a
  real, pickable `Renderable` entity (at the camera pivot, scaled to orbit
  distance) with a matching `PickGeometry`. So generated shapes are themselves
  selectable and can be fed to the geometry operators above.
- **Tree-view / scene outliner (`TreeView`):** a draggable overlay widget built
  on the same dynamic-VB + font-atlas quad pattern as the FPS widget. Lists the
  world's drawable entities under collapsible group headers (Meshes / Point
  Clouds); rows are rebuilt from the world each frame (the component holds only UI
  state). `treeViewInputSystem` handles title-bar drag, wheel scroll, group
  expand/collapse, and row clicks — a click writes `Renderable::selected` (plain =
  replace, Ctrl = toggle), so it is two-way synced with the viewport silhouette
  selection. `renderSystem` draws it as an overlay (title bar drawn last / higher
  z so scrolled-up rows vanish under it). Created in appOrange's `main.cpp`
  alongside the FPS widget; `kTreeQuads` (systems.cpp) must match the app's index
  capacity.
- **Text/UI:** gizmo labels and FPS/controls widgets are rasterized with
  stb_truetype (loads `C:/Windows/Fonts/malgun.ttf` on Windows) into RGBA atlases
  drawn as textured overlay quads via `IRenderer::beginOverlay`. Overlay layers
  need distinct z or depth-test rejects coplanar quads. stb pack glyph quad size
  must use `xoff2-xoff` (not `x1-x0`) or oversampled glyphs render 2× too big.
- **Screenshot:** press `C` to save `orange_capture.png` next to the exe
  (`IRenderer::readPixels`, GL only). Useful to verify rendering visually.

[EnTT]: https://github.com/skypjack/entt
[SDL3]: https://github.com/libsdl-org/SDL
