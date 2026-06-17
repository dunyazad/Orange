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
app/sandbox/         demo app: three spinning cubes via ECS (also hosts font/UI impl)
```

## Build & run

Requires CMake ≥ 3.21 and a C++17 compiler. SDL3, EnTT, and stb are fetched via
FetchContent on first configure (needs network). All binaries land in the bin
output dir so the app finds plugins next to itself.

Generic (per README):
```sh
cmake -S . -B build
cmake --build build --config Debug
./build/bin/sandbox            # OpenGL (default)
./build/bin/sandbox --vulkan   # Vulkan
```

### ⚠️ This Windows machine (D:\Library\Orange)

The shell has `cl`/`ninja` on PATH but **NOT** the full MSVC env (no
`INCLUDE`/`VCINSTALLDIR`), so **Ninja fails**. Use the Visual Studio generator:

```sh
cmake -S D:/Library/Orange -B D:/Library/Orange/build -G "Visual Studio 16 2019" -A x64
cmake --build D:/Library/Orange/build --config Debug
```

- Vulkan SDK is at `C:\VulkanSDK\1.3.283.0`, so `render_vk` builds here.
- Binaries land in `build/bin/Debug`. Run `sandbox.exe` (GL) or
  `sandbox.exe --vulkan`.

## Key facts to keep in mind

- **Plugin ABI is v4.** Changing the C contract means bumping
  `ORANGE_PLUGIN_ABI_VERSION` and updating both backends.
- **Buffers are two-layer:** handle + byte-size based ABI in `render_api`, and a
  type-safe `core::Buffer<T>` (VertexBuffer/IndexBuffer/UniformBuffer) in
  `engine/core/include/orange/core/buffer.h` that wraps it with RAII. App code
  uses the template layer.
- **Vulkan shaders:** `plugins/render_vk/shaders/*.{vert,frag}` are compiled to
  SPIR-V headers in `src/generated/` (CMake regenerates on change via the Vulkan
  SDK's glsl compiler). GL→VK clip-space correction (Y flip + depth 0..1) is
  applied so both backends show the identical scene.
- **No gimbal lock:** `Transform::orientation` is a unit quaternion; the camera
  is a quaternion arcball trackball (`CameraManipulator`). Any camera controller
  is just a system that writes a `Transform`.
- **Text/UI:** gizmo labels and FPS/controls widgets are rasterized with
  stb_truetype (loads `C:/Windows/Fonts/malgun.ttf` on Windows) into RGBA atlases
  drawn as textured overlay quads via `IRenderer::beginOverlay`. Overlay layers
  need distinct z or depth-test rejects coplanar quads. stb pack glyph quad size
  must use `xoff2-xoff` (not `x1-x0`) or oversampled glyphs render 2× too big.
- **Screenshot:** press `C` to save `orange_capture.png` next to the exe
  (`IRenderer::readPixels`, GL only). Useful to verify rendering visually.

[EnTT]: https://github.com/skypjack/entt
[SDL3]: https://github.com/libsdl-org/SDL
