#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <entt/entt.hpp>

#include "orange/ecs/components.h"
#include "orange/render/renderer.h"

// Application-level undo/redo. Document mutations (spawn/delete an entity,
// point removal, per-mesh draw/color mode changes) push an UndoOp with
// symmetric undo/redo closures capturing the data they need -- CPU vertex
// snapshots for GPU rebuilds. View state (camera, selection, grid, lighting,
// point size) is deliberately NOT recorded.
//
// Entity identity across destroy/recreate cycles: a rebuilt entity gets a new
// entt id, so spawn/delete ops share one EntitySnapshot per logical object
// (attached to the entity as an UndoRef component); every rebuild updates
// snapshot->entity, keeping later ops on the same object pointed right.
// Draw/color ops hold raw entity ids and simply no-op if the id went stale.

namespace orange::ecs {

struct UndoOp {
    std::string label;
    std::function<void(entt::registry&, render::IRenderer&)> undo;
    std::function<void(entt::registry&, render::IRenderer&)> redo;
};

// Everything needed to rebuild a drawable entity (GPU buffers + mesh +
// components) after an undo-of-delete / redo-of-spawn.
struct EntitySnapshot {
    entt::entity entity = entt::null;  // current incarnation (updated on rebuild)
    std::vector<render::Vertex> vertices;
    std::vector<uint32_t>       indices;  // triangle list; empty + pointCloud => Points
    Transform      transform;
    core::DrawMode drawMode   = core::DrawMode::Solid;
    uint32_t       colorMode  = 0;
    bool           pointCloud = false;
};
using EntitySnapshotPtr = std::shared_ptr<EntitySnapshot>;

// Component tying an entity to its shared snapshot (see header comment).
struct UndoRef {
    EntitySnapshotPtr snap;
};

// Registry-ctx singleton holding the two stacks.
struct UndoStack {
    std::vector<UndoOp> done;    // undo candidates (most recent last)
    std::vector<UndoOp> undone;  // redo candidates
    static constexpr size_t kMax = 64;  // vertex snapshots are big; cap memory

    void push(UndoOp op) {
        done.push_back(std::move(op));
        if (done.size() > kMax) done.erase(done.begin());
        undone.clear();  // a fresh edit invalidates the redo branch
    }
};

UndoStack& undoStack(entt::registry& world);

// Capture `e` into (or refresh) its shared snapshot. Returns null when the
// entity can't be rebuilt (it has no VertexSource CPU copy).
EntitySnapshotPtr captureEntity(entt::registry& world, entt::entity e);

// Recreate the GPU buffers/mesh + components from the snapshot; updates
// snap->entity to the new incarnation.
void rebuildEntity(entt::registry& world, render::IRenderer& renderer,
                   const EntitySnapshotPtr& snap);

// Record "entity e was just spawned" (undo destroys it, redo rebuilds it).
void pushSpawnOp(entt::registry& world, entt::entity e, const char* label);

// Record "these entities are about to be deleted" (undo rebuilds them, redo
// re-deletes). Call BEFORE destroying them; the caller still does the destroy.
void pushDeleteOp(entt::registry& world, const std::vector<entt::entity>& dead,
                  const char* label);

// Pop + execute one op. Returns false when the respective stack is empty.
bool undoLast(entt::registry& world, render::IRenderer& renderer);
bool redoLast(entt::registry& world, render::IRenderer& renderer);

}  // namespace orange::ecs
