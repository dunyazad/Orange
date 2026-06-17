# Orange

An ECS-driven application with a **plugin-based rendering engine**. Render
backends (OpenGL, Vulkan) are compiled as separate shared libraries and loaded
at runtime through a stable C ABI — so a backend is hot-swappable and the app
core never links against OpenGL or Vulkan directly.

- **ECS, not a scene graph.** World state is a flat set of entities +
  components ([EnTT](https://github.com/skypjack/entt)). Systems transform that
  data; the render system is the bridge from ECS to draw calls.
- **Plugin boundary.** The app talks only to `orange::render::IRenderer`
  (`engine/render_api`). Backends implement it and export three C symbols.
- **Cross-platform by design.** [SDL3](https://github.com/libsdl-org/SDL)
  handles window/input/GL-context/Vulkan-surface on desktop **and** mobile, so
  macOS / iOS / Android are an additive step, not a rewrite.

## Architecture

```
app/appOrange ───────────┐  loads at runtime (dlopen / LoadLibrary)
                         │
engine/core  ── ECS (EnTT), window (SDL3), plugin loader, app loop
   │  depends on
   ▼
engine/render_api  ── IRenderer interface + C ABI contract (no GL/VK/SDL)
   ▲  implements
   │
plugins/render_gl  ── OpenGL 3.3 backend  (render_gl.dll / librender_gl.so)
plugins/render_vk  ── Vulkan backend      (render_vk.dll / librender_vk.so)
```

The plugin contract lives in `engine/render_api/include/orange/render/`:

| Symbol                     | Purpose                          |
|----------------------------|----------------------------------|
| `orangeRenderPluginInfo`   | backend type + ABI version       |
| `orangeRenderCreate`       | construct an `IRenderer`          |
| `orangeRenderDestroy`      | destroy it                       |

The host resolves these by name, checks `ORANGE_PLUGIN_ABI_VERSION`, then drives
the renderer through the abstract interface only.

### GPU resources

Resources come in two layers:

**1. Handle-based ABI (`engine/render_api`).** A template can't cross a C ABI /
vtable boundary, so the plugin contract is handle + byte-size based and backend
neutral:

```cpp
BufferHandle vbo = renderer->createBuffer({BufferType::Vertex, BufferUsage::Static, data, size});
renderer->updateBuffer(vbo, newData, size);   // sub-range upload (Dynamic buffers)
renderer->destroyBuffer(vbo);
```

**2. Type-safe template (`engine/core/buffer.h`).** A host-side `Buffer<T>`
wraps the handle API: the element type is a template parameter, size is
`count * sizeof(T)`, `update()` takes a `const T*`, and the GPU buffer is freed
by RAII. This is the layer app code uses:

```cpp
core::VertexBuffer<Vertex> vbo(renderer, vertices);  // std::vector / list / ptr+count
core::IndexBuffer          ibo(renderer, indices);   // element type fixed to uint32_t
core::UniformBuffer<Camera> ubo(renderer, &cam, 1, BufferUsage::Dynamic);
```

A **mesh** is a thin composition over buffers + a `VertexLayout` (it owns only
the binding state, e.g. a GL VAO — not the buffers):

```cpp
MeshHandle mesh = renderer->createMesh({vbo.handle(), ibo.handle(), Vertex::layout(),
                                        vbo.count(), ibo.count()});
```

Each backend maps the handle to its native object: GL buffer name (`glGenBuffers`)
or `VkBuffer` + `VkDeviceMemory`. Vertex/index/uniform buffer types are all
supported.

**Textures.** `createTexture(TextureDesc)` uploads an RGBA8 image and returns a
`TextureHandle`; a `DrawItem` carries an optional `texture` (invalid → a 1×1
white fallback, so untextured draws keep their vertex color). The shared shader
samples `texture(uTex, uv) * vertexColor` with alpha blending. GL uses a texture
object + sampler state; Vulkan uses a staged image upload (layout transitions)
plus a combined-image-sampler descriptor set per texture. The axis-gizmo labels
are drawn this way — a text atlas sampled by one quad per face.

## Layout

```
engine/render_api/   interface-only contract (header library)
engine/core/         platform layer + ECS (components, systems, app loop)
plugins/render_gl/   OpenGL backend plugin
plugins/render_vk/   Vulkan backend plugin (built only if the Vulkan SDK exists)
app/appOrange/       demo app: spinning cubes via ECS
```

## Build

Requires CMake ≥ 3.21 and a C++17 compiler. SDL3 and EnTT are fetched
automatically on first configure (needs network access). The Vulkan plugin is
built only when a Vulkan SDK is found.

```sh
cmake -S . -B build
cmake --build build --config Debug
```

Binaries (app + plugins) land in `build/bin/`.

## Run

```sh
./build/bin/appOrange            # OpenGL (default)
./build/bin/appOrange --vulkan   # Vulkan (clears + presents; mesh draw is WIP)
```

### Controls

A trackball camera (`CameraManipulator` component) orbits the scene:

| Input               | Action          |
|---------------------|-----------------|
| Left-drag           | Orbit (quaternion trackball — tumbles freely, no gimbal lock) |
| Mouse wheel         | Zoom (distance) |
| Middle/right-drag   | Pan (move target) |
| ESC / close window  | Quit            |

Orientation is accumulated as a unit quaternion (`Transform::orientation`), so
there is no gimbal lock. `Spin` and the camera both integrate rotations on the
quaternion, never on Euler angles.

### Axis gizmo (ViewCube)

A face-colored cube (with X/Y/Z face labels) in the top-right corner
(`AxisGizmo` component) shows the world orientation as the camera sees it.

- **Hover** a face/edge/corner → a highlight patch that conforms to the cube
  faces (one face cell, two for an edge, three for a corner).
- **Click** a face/edge/corner → smoothly snap the camera to that view
  (quaternion `slerp`, ~0.45 s ease).
- A **4-sector ring** behind the cube (right/top/left/bottom) highlights on hover
  and, on click, rotates the camera 90° in that direction.

It is drawn as an overlay sub-pass (`IRenderer::beginOverlay` — corner viewport,
own orthographic view/proj, depth cleared so it sits on top). The cube ray-pick
classifies the hit into a `{-1,0,1}³` direction (one axis → face, two → edge,
three → corner); a miss falls through to a ring annulus/sector test. The cube,
labels, and highlight share the gizmo's orientation; the ring is screen-aligned.

Mouse events flow `SDL → Application::input_ (core::Input) → cameraManipulatorSystem`,
which writes the camera entity's `Transform`. `renderSystem` derives the view
matrix from that Transform's orientation, so any camera controller is just a
system that writes a `Transform` — no special camera path.

## Current status

- **Buffer abstraction:** complete on both backends — real GL buffers and real
  `VkBuffer` + device memory, with create/update/destroy and vertex/index/
  uniform types.
- **OpenGL backend:** complete — renders ECS mesh entities from buffer objects
  (indexed draw, depth test, per-frame view/projection).
- **Vulkan backend:** complete for the same scene — graphics pipeline built from
  the mesh `VertexLayout`, SPIR-V shaders (push-constant viewProj/model), depth
  buffer, dynamic viewport/scissor, indexed draw, and swapchain recreation on
  resize. GL→Vulkan clip-space correction (Y flip + depth 0..1) is applied so
  both backends show the identical scene.

Both backends render the same three spinning cubes; switch with `--gl` /
`--vulkan`.

## Extending to macOS / iOS / Android

Nothing in `engine/core` or the plugin contract is desktop-specific:

- **macOS:** builds as-is; OpenGL is deprecated there, so prefer the Vulkan
  plugin via MoltenVK (set `VULKAN_SDK`).
- **iOS / Android:** SDL3 supports both. Use the OpenGL **ES** path or Vulkan
  (MoltenVK on iOS). The plugin loader already uses `dlopen`; on mobile you can
  also link a backend statically and register it without the dynamic loader —
  the `IRenderer` interface stays identical.
