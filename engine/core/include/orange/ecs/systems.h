#pragma once

#include <entt/entt.hpp>

#include "orange/core/input.h"
#include "orange/render/renderer.h"

namespace orange::ecs {

// Advances all Spin components (rotates their Transform).
void spinSystem(entt::registry& world, float dt);

// Trackball camera control: reads mouse input and updates the Transform of
// every entity that has (Transform, Camera, CameraManipulator).
void cameraManipulatorSystem(entt::registry& world, const core::Input& input,
                             float dt);

// FPS widget: updates the FPS history and handles dragging the widget. Sets
// input.captured while dragging so the camera doesn't orbit at the same time.
void fpsWidgetInputSystem(entt::registry& world, core::Input& input, float dt,
                          uint32_t viewportW, uint32_t viewportH);

// Camera controls panel: positions the panel under the gizmo and handles clicks
// on its buttons (toggle projection, FOV/size +/-). Sets input.captured on use.
void cameraControlsInputSystem(entt::registry& world, core::Input& input, float dt,
                               uint32_t viewportW, uint32_t viewportH);

// Axis gizmo input: each frame, ray-picks the gizmo cube under the cursor to
// update the hover region; on click, classifies the hit as face/edge/corner and
// starts a smooth camera snap toward that view (plus a click-flash). Runs before
// cameraManipulatorSystem.
void axisGizmoInputSystem(entt::registry& world, core::Input& input,
                          float dt, uint32_t viewportW, uint32_t viewportH);

// Top menu bar: handles opening/closing the File menu and clicking its items.
// Sets input.captured while the pointer is over the bar or open dropdown (so the
// scene neither orbits nor picks), and raises MenuBar::requestOpenFile when
// "Open..." is chosen. Runs first so it gets first crack at the click.
void menuBarInputSystem(entt::registry& world, core::Input& input,
                        uint32_t viewportW, uint32_t viewportH);

// Picking: on a fresh left-click (not consumed by a UI widget), casts a ray
// from the primary camera through the cursor, finds the nearest (Transform,
// Renderable) hit via ray-vs-local-AABB, and marks it Renderable::selected
// (single-select). Runs after cameraManipulatorSystem so the camera is current.
void pickingSystem(entt::registry& world, const core::Input& input,
                   uint32_t viewportW, uint32_t viewportH);

// True if entity `e` (Transform + Renderable) projects inside the primary
// camera's viewport — i.e. it is actually visible on screen. Used to stop
// off-screen meshes from being selected (e.g. by Ctrl+A).
bool entityVisibleOnScreen(entt::registry& world, entt::entity e,
                           uint32_t viewportW, uint32_t viewportH);

// Runs the active point-cloud processing mode (modes::ModeState in ctx) on the
// input cloud (modes::ModeInput in ctx) and emits its visualization into the
// debug-draw accumulator. Recomputes only when the selected mode changes; no-op
// if no ModeInput is present. Call before renderSystem.
void processingModeSystem(entt::registry& world);

// Gathers the primary camera + all (Transform, MeshRenderer) entities and
// issues draw calls through the renderer. This is the ECS -> render bridge.
void renderSystem(entt::registry& world, render::IRenderer& renderer,
                  uint32_t viewportW, uint32_t viewportH);

} // namespace orange::ecs
