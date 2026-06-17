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
void cameraControlsInputSystem(entt::registry& world, core::Input& input,
                               uint32_t viewportW, uint32_t viewportH);

// Axis gizmo input: each frame, ray-picks the gizmo cube under the cursor to
// update the hover region; on click, classifies the hit as face/edge/corner and
// starts a smooth camera snap toward that view (plus a click-flash). Runs before
// cameraManipulatorSystem.
void axisGizmoInputSystem(entt::registry& world, const core::Input& input,
                          float dt, uint32_t viewportW, uint32_t viewportH);

// Gathers the primary camera + all (Transform, MeshRenderer) entities and
// issues draw calls through the renderer. This is the ECS -> render bridge.
void renderSystem(entt::registry& world, render::IRenderer& renderer,
                  uint32_t viewportW, uint32_t viewportH);

} // namespace orange::ecs
