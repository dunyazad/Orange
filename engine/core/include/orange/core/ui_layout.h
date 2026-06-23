#pragma once

#include <string>

#include <entt/entt.hpp>

namespace orange::core {

// Persist and restore the positions of the freely-draggable overlay widgets
// (FpsWidget + TreeView) to a small text file, so each launch reopens them where
// they were last left. Anchored panels (camera controls, cross-section, gizmo)
// derive their position from the gizmo each frame and are not stored.
//
// Call loadWidgetLayout once after the widgets are created (before the run loop),
// and saveWidgetLayout on exit (after the run loop returns). A missing/corrupt
// file is ignored -- the components keep their defaults.
void saveWidgetLayout(const entt::registry& world, const std::string& path);
void loadWidgetLayout(entt::registry& world, const std::string& path);

} // namespace orange::core
