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

## Layout

```
engine/render_api/   interface-only C-ABI contract (header library) — no GL/VK/SDL
engine/core/         platform layer + ECS (components, systems, app loop, window)
plugins/render_gl/   OpenGL 3.3 backend plugin
plugins/render_vk/   Vulkan backend plugin (built only if a Vulkan SDK is found)
app/appOrange/       demo app: three spinning cubes via ECS (also hosts font/UI impl)
```

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

- **Plugin ABI is v7.** Changing the C contract means bumping
  `ORANGE_PLUGIN_ABI_VERSION` and updating both backends. v6 added
  `IRenderer::setVsync(bool)` (GL flips the swap interval; VK recreates the
  swapchain with a FIFO / immediate-mailbox present mode). v7 added
  `IRenderer::drawGrid()`.
- **Infinite grid:** `renderSystem` calls `IRenderer::drawGrid()` after the scene
  submits (before the gizmo overlay). Each backend renders a vertex-less
  full-screen pass (GL: inline shader + empty VAO; VK: a second pipeline +
  `shaders/grid.{vert,frag}` → SPIR-V) that ray-casts each pixel onto the world
  y=0 plane and draws `fwidth`-based AA grid lines with distance fade and correct
  depth (so the scene occludes it). Each plugin inverts its own clip-corrected
  `viewProj` for the unprojection (the host doesn't, since VK's correction is
  applied plugin-side).
- **Buffers are two-layer:** handle + byte-size based ABI in `render_api`, and a
  type-safe `core::Buffer<T>` (VertexBuffer/IndexBuffer/UniformBuffer) in
  `engine/core/include/orange/core/buffer.h` that wraps it with RAII. App code
  uses the template layer.
- **Vulkan shaders:** `plugins/render_vk/shaders/*.{vert,frag}` are compiled to
  SPIR-V headers in `src/generated/` (CMake regenerates on change via the Vulkan
  SDK's glsl compiler). GL→VK clip-space correction (Y flip + depth 0..1) is
  applied so both backends show the identical scene.
- **Math is Eigen-backed.** `orange/core/math.h` aliases `Vec3/Vec4/Mat4/Quat`
  to `Eigen::Vector3f/Vector4f/Matrix4f/Quaternionf` (column-major, GL convention)
  and keeps thin graphics helpers (`perspective`/`ortho`/`lookAt`/`quatAxisAngle`/
  `toMat4`/...). Eigen is fetched via FetchContent (find_package first). The build
  defines `EIGEN_MAX_ALIGN_BYTES=0` so Eigen members are safe inside EnTT pools.
  Interop with the render ABI's `float[16]`/`float[3]` is via `.data()`. Use
  `.x()/.y()/.z()` (methods), not `.x`, and construct with `Vec3(a,b,c)` /
  `Quat::Identity()` (the default Eigen ctor leaves values uninitialized).
- **No gimbal lock:** `Transform::orientation` is a unit quaternion; the camera
  is a quaternion arcball trackball (`CameraManipulator`). Any camera controller
  is just a system that writes a `Transform`.
- **Input scheme:** right-drag orbits, middle-drag pans, wheel zooms, and
  **left-click picks** the nearest `Renderable` via `pickingSystem` (builds a
  world ray from the camera and ray-tests each entity's local AABB; the hit gets
  `Renderable::selected`). UI widgets set `Input::captured` to suppress picking.
- **Text/UI:** gizmo labels and FPS/controls widgets are rasterized with
  stb_truetype (loads `C:/Windows/Fonts/malgun.ttf` on Windows) into RGBA atlases
  drawn as textured overlay quads via `IRenderer::beginOverlay`. Overlay layers
  need distinct z or depth-test rejects coplanar quads. stb pack glyph quad size
  must use `xoff2-xoff` (not `x1-x0`) or oversampled glyphs render 2× too big.
- **Screenshot:** press `C` to save `orange_capture.png` next to the exe
  (`IRenderer::readPixels`, GL only). Useful to verify rendering visually.

[EnTT]: https://github.com/skypjack/entt
[SDL3]: https://github.com/libsdl-org/SDL
