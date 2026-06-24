#pragma once

#include <vector>

#include <entt/entt.hpp>

#include "orange/core/input.h"
#include "orange/ecs/components.h"
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

// Tree-view (scene outliner) widget: positions the panel, handles title-bar drag,
// scroll-wheel, group expand/collapse, and row clicks (which select the entity by
// writing Renderable::selected). Sets input.captured over the panel so clicks
// don't also pick in the viewport.
void treeViewInputSystem(entt::registry& world, core::Input& input,
                         uint32_t viewportW, uint32_t viewportH);

// Camera controls panel: positions the panel under the gizmo and handles clicks
// on its buttons (toggle projection, FOV/size +/-). Sets input.captured on use.
void cameraControlsInputSystem(entt::registry& world, core::Input& input, float dt,
                               uint32_t viewportW, uint32_t viewportH);

// Cross-section panel: positions it under the camera-controls panel and handles
// the enable toggle, axis cycle (X/Y/Z), flip toggle, and slider drag that moves
// the cut plane. Sets input.captured while the pointer is over the panel.
void crossSectionInputSystem(entt::registry& world, core::Input& input,
                             uint32_t viewportW, uint32_t viewportH);

// Poisson reconstruction dialog: handles the parameter sliders and the
// "Reconstruct" button (which writes geometry::PoissonParams into the ctx and
// activates the Poisson processing mode). Sets input.captured over the panel.
// No-op while the dialog is hidden.
void poissonDialogInputSystem(entt::registry& world, core::Input& input,
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

// Builds the default appOrange menu set (File / View / Render / Draw / Select /
// Points / Help) wired to the MenuAction ids. Assign the result to
// MenuBar::menus at setup. The titles/items map 1:1 to the keyboard shortcuts and
// the dispatch in Application::applyMenuAction.
std::vector<Menu> defaultAppMenus();

// Left selection toolbar: hit-tests the mode buttons and writes the chosen value
// into the ctx SelectionMode. Sets input.captured over the bar. Runs before
// picking so a button click never falls through to the scene.
void selectionToolbarInputSystem(entt::registry& world, core::Input& input,
                                 uint32_t viewportW, uint32_t viewportH);

// The default toolbar button set (target / action / filter / modifier groups).
std::vector<ToolbarButton> defaultSelectionToolbar();

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
// selected entity's cloud, on a BACKGROUND thread so the main loop never blocks;
// emits the last result into the debug-draw accumulator each frame. Recomputes
// only when the active mode or selection changes. Call before renderSystem.
void processingModeSystem(entt::registry& world);

// If a processing-mode computation is running in the background, returns true and
// fills `outProgress` (0..1) + `outName` (mode label) for a UI status line; false
// when idle. Lets the app show e.g. "Smooth 42%".
bool processingModeProgress(entt::registry& world, float& outProgress, std::string& outName);

// Same, for a background SpatialViz wireframe build (the spatial-structure overlay
// is built off-thread too). Returns true with progress + "Spatial" while building.
bool spatialVizProgress(entt::registry& world, float& outProgress, std::string& outName);

// Gathers the primary camera + all (Transform, MeshRenderer) entities and
// issues draw calls through the renderer. This is the ECS -> render bridge.
void renderSystem(entt::registry& world, render::IRenderer& renderer,
                  uint32_t viewportW, uint32_t viewportH);

} // namespace orange::ecs
