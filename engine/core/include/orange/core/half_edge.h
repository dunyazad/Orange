#pragma once

// Half-edge (DCEL) surface mesh for O(1) one-ring / adjacency traversal.
//
// Built from an indexed triangle list. Positions are WELDED (coincident vertices
// merged) so a triangle-soup mesh -- e.g. a binary STL, where every triangle
// carries its own 3 vertices and nothing is shared -- still forms a connected,
// shared-vertex topology that can be walked. Triangles only.
//
// ASCII comments only (pipeline project rule).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace orange::geometry {

class HalfEdgeMesh {
public:
    struct HalfEdge {
        uint32_t origin = 0;   // vertex this half-edge starts from
        int32_t  twin   = -1;  // opposite half-edge (-1 on a boundary edge)
        uint32_t next   = 0;   // next half-edge around the same triangle (CCW)
        uint32_t prev   = 0;   // previous half-edge around the same triangle
        uint32_t face   = 0;   // incident triangle index
    };

    // Build from welded triangles. `weldEps <= 0` picks an automatic epsilon from
    // the bbox diagonal. Returns false on empty/degenerate input.
    bool build(const std::vector<Eigen::Vector3f>& positions,
               const std::vector<uint32_t>& indices, float weldEps = -1.0f);

    size_t vertexCount() const { return vertices_.size(); }
    size_t halfEdgeCount() const { return halfEdges_.size(); }
    const Eigen::Vector3f& position(uint32_t v) const { return vertices_[v]; }
    const std::vector<Eigen::Vector3f>& positions() const { return vertices_; }
    const std::vector<HalfEdge>& halfEdges() const { return halfEdges_; }

    // True if vertex v lies on a mesh boundary (some incident edge has no twin).
    bool isBoundary(uint32_t v) const;

    // One-ring neighbour vertices of v (welded indexing), ordered around v. On a
    // boundary the open fan is returned (both directions, no wrap). Empty if v is
    // isolated.
    std::vector<uint32_t> oneRing(uint32_t v) const;

    // Append the one-ring of v into `out` (cleared first). Avoids per-call
    // allocation in tight loops over every vertex.
    void oneRing(uint32_t v, std::vector<uint32_t>& out) const;

    // Map an original (pre-weld) vertex index to its welded index.
    uint32_t welded(uint32_t original) const { return remap_[original]; }

private:
    std::vector<Eigen::Vector3f> vertices_;   // welded positions
    std::vector<HalfEdge>        halfEdges_;
    std::vector<int32_t>         vertHalf_;    // one outgoing half-edge per vertex (-1 if none)
    std::vector<uint32_t>        remap_;       // original -> welded index
};

} // namespace orange::geometry
