#pragma once

#include <vector>

#include <Eigen/Core>

#include "orange/render/types.h"

// Immediate-mode debug drawing. A global, per-frame triangle accumulator: call
// the add* helpers from anywhere during the frame, and the renderSystem uploads
// everything as one dynamic mesh and clears it for the next frame.
//
// Inspired by Elements/Helium VisualDebugging, but adapted to Orange's mesh-only
// renderer: there is no line primitive, so lines/wire boxes are emitted as thin
// triangle tubes. Colors map straight to the unlit vertex color.

namespace orange::debug {

class DebugDraw {
public:
    DebugDraw() = default;

    // The global per-frame accumulator the renderSystem uploads + clears. Build
    // into a local DebugDraw instead when you want to cache geometry across frames.
    static DebugDraw& instance() {
        static DebugDraw d;
        return d;
    }

    void clear() { vertices_.clear(); }
    const std::vector<render::Vertex>& vertices() const { return vertices_; }
    bool empty() const { return vertices_.empty(); }

    // Append a precomputed vertex list (e.g. a cached mode result) verbatim.
    void addRaw(const std::vector<render::Vertex>& verts) {
        vertices_.insert(vertices_.end(), verts.begin(), verts.end());
    }

    // A point drawn as a tiny tetrahedron (visible from any angle, 4 triangles).
    void addPoint(const Eigen::Vector3f& p, const Eigen::Vector3f& color, float size = 0.01f) {
        Eigen::Vector3f a = p + Eigen::Vector3f(0, size, 0);
        Eigen::Vector3f b = p + Eigen::Vector3f(-size, -size * 0.5f, -size * 0.5f);
        Eigen::Vector3f c = p + Eigen::Vector3f(size, -size * 0.5f, -size * 0.5f);
        Eigen::Vector3f d = p + Eigen::Vector3f(0, -size * 0.5f, size);
        addTriangle(a, b, c, color);
        addTriangle(a, c, d, color);
        addTriangle(a, d, b, color);
        addTriangle(b, d, c, color);
    }

    void addTriangle(const Eigen::Vector3f& a, const Eigen::Vector3f& b,
                     const Eigen::Vector3f& c, const Eigen::Vector3f& color) {
        pushVert(a, color);
        pushVert(b, color);
        pushVert(c, color);
    }

    // A quad a-b-c-d (two triangles), single-sided.
    void addQuad(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c,
                 const Eigen::Vector3f& d, const Eigen::Vector3f& color) {
        addTriangle(a, b, c, color);
        addTriangle(a, c, d, color);
    }

    // Line a->b as a thin square-section tube so it is visible from any angle.
    void addLine(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& color,
                 float thickness = 0.01f) {
        Eigen::Vector3f dir = (b - a);
        float len = dir.norm();
        if (len < 1e-8f) return;
        dir /= len;
        Eigen::Vector3f up = std::abs(dir.x()) < 0.9f ? Eigen::Vector3f(1, 0, 0)
                                                      : Eigen::Vector3f(0, 1, 0);
        Eigen::Vector3f u = dir.cross(up).normalized() * thickness;
        Eigen::Vector3f v = dir.cross(u).normalized() * thickness;
        Eigen::Vector3f a0 = a - u - v, a1 = a + u - v, a2 = a + u + v, a3 = a - u + v;
        Eigen::Vector3f b0 = b - u - v, b1 = b + u - v, b2 = b + u + v, b3 = b - u + v;
        addQuad(a0, a1, b1, b0, color);  // 4 long faces
        addQuad(a1, a2, b2, b1, color);
        addQuad(a2, a3, b3, b2, color);
        addQuad(a3, a0, b0, b3, color);
    }

    // Solid axis-aligned box.
    void addBox(const Eigen::Vector3f& mn, const Eigen::Vector3f& mx, const Eigen::Vector3f& color) {
        Eigen::Vector3f c[8] = {
            {mn.x(), mn.y(), mn.z()}, {mx.x(), mn.y(), mn.z()}, {mx.x(), mx.y(), mn.z()},
            {mn.x(), mx.y(), mn.z()}, {mn.x(), mn.y(), mx.z()}, {mx.x(), mn.y(), mx.z()},
            {mx.x(), mx.y(), mx.z()}, {mn.x(), mx.y(), mx.z()}};
        addQuad(c[0], c[1], c[2], c[3], color);  // -Z
        addQuad(c[5], c[4], c[7], c[6], color);  // +Z
        addQuad(c[4], c[0], c[3], c[7], color);  // -X
        addQuad(c[1], c[5], c[6], c[2], color);  // +X
        addQuad(c[4], c[5], c[1], c[0], color);  // -Y
        addQuad(c[3], c[2], c[6], c[7], color);  // +Y
    }

    // Wireframe box: 12 edges as thin tubes.
    void addWireBox(const Eigen::Vector3f& mn, const Eigen::Vector3f& mx,
                    const Eigen::Vector3f& color, float thickness = 0.01f) {
        Eigen::Vector3f c[8] = {
            {mn.x(), mn.y(), mn.z()}, {mx.x(), mn.y(), mn.z()}, {mx.x(), mx.y(), mn.z()},
            {mn.x(), mx.y(), mn.z()}, {mn.x(), mn.y(), mx.z()}, {mx.x(), mn.y(), mx.z()},
            {mx.x(), mx.y(), mx.z()}, {mn.x(), mx.y(), mx.z()}};
        const int e[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                              {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (auto& edge : e) addLine(c[edge[0]], c[edge[1]], color, thickness);
    }

    // Solid lat/long sphere.
    void addSphere(const Eigen::Vector3f& center, float radius, const Eigen::Vector3f& color,
                   int segments = 12) {
        if (segments < 3) segments = 3;
        const float pi = 3.14159265358979323846f;
        auto at = [&](int i, int j) {
            float theta = pi * float(i) / float(segments);        // 0..pi  (lat)
            float phi = 2.0f * pi * float(j) / float(segments);   // 0..2pi (lon)
            return center + radius * Eigen::Vector3f(std::sin(theta) * std::cos(phi),
                                                     std::cos(theta),
                                                     std::sin(theta) * std::sin(phi));
        };
        for (int i = 0; i < segments; ++i)
            for (int j = 0; j < segments; ++j)
                addQuad(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1), color);
    }

private:
    void pushVert(const Eigen::Vector3f& p, const Eigen::Vector3f& color) {
        render::Vertex v;
        v.position[0] = p.x(); v.position[1] = p.y(); v.position[2] = p.z();
        v.color[0] = color.x(); v.color[1] = color.y(); v.color[2] = color.z();
        v.uv[0] = 0.0f; v.uv[1] = 0.0f;
        vertices_.push_back(v);
    }

    std::vector<render::Vertex> vertices_;
};

} // namespace orange::debug
