#include "orange/core/primitives.h"

#include <algorithm>
#include <cmath>

#include <Eigen/Geometry>

namespace orange::geometry {
namespace {

constexpr float kPi = 3.14159265358979323846f;

// Append a triangle with a flat (face) normal.
void addFlat(std::vector<Triangle>& out, const Eigen::Vector3f& a, const Eigen::Vector3f& b,
             const Eigen::Vector3f& c, const Eigen::Vector3f& color) {
    Eigen::Vector3f n = (b - a).cross(c - a);
    n = n.squaredNorm() > 1e-20f ? n.normalized() : Eigen::Vector3f::UnitY();
    Triangle t;
    t.v[0] = a; t.v[1] = b; t.v[2] = c;
    t.n[0] = t.n[1] = t.n[2] = n;
    t.c[0] = t.c[1] = t.c[2] = color;
    out.push_back(t);
}

// Append a triangle with explicit per-vertex (smooth) normals.
void addSmooth(std::vector<Triangle>& out, const Eigen::Vector3f& a, const Eigen::Vector3f& b,
               const Eigen::Vector3f& c, const Eigen::Vector3f& na, const Eigen::Vector3f& nb,
               const Eigen::Vector3f& nc, const Eigen::Vector3f& color) {
    Triangle t;
    t.v[0] = a; t.v[1] = b; t.v[2] = c;
    t.n[0] = na; t.n[1] = nb; t.n[2] = nc;
    t.c[0] = t.c[1] = t.c[2] = color;
    out.push_back(t);
}

// Flat quad (a,b,c,d CCW) as two triangles.
void addQuad(std::vector<Triangle>& out, const Eigen::Vector3f& a, const Eigen::Vector3f& b,
             const Eigen::Vector3f& c, const Eigen::Vector3f& d, const Eigen::Vector3f& color) {
    addFlat(out, a, b, c, color);
    addFlat(out, a, c, d, color);
}

} // namespace

std::vector<Triangle> buildPlane(float size, const Eigen::Vector3f& color) {
    float h = size * 0.5f;
    std::vector<Triangle> out;
    addQuad(out, {-h, 0, -h}, {-h, 0, h}, {h, 0, h}, {h, 0, -h}, color);
    return out;
}

std::vector<Triangle> buildBox(const Eigen::Vector3f& size, const Eigen::Vector3f& color) {
    Eigen::Vector3f h = size * 0.5f;
    std::vector<Triangle> out;
    Eigen::Vector3f p[8] = {
        {-h.x(), -h.y(), -h.z()}, {h.x(), -h.y(), -h.z()},
        {h.x(), h.y(), -h.z()},   {-h.x(), h.y(), -h.z()},
        {-h.x(), -h.y(), h.z()},  {h.x(), -h.y(), h.z()},
        {h.x(), h.y(), h.z()},    {-h.x(), h.y(), h.z()}};
    addQuad(out, p[0], p[3], p[2], p[1], color);  // -Z
    addQuad(out, p[4], p[5], p[6], p[7], color);  // +Z
    addQuad(out, p[0], p[4], p[7], p[3], color);  // -X
    addQuad(out, p[1], p[2], p[6], p[5], color);  // +X
    addQuad(out, p[0], p[1], p[5], p[4], color);  // -Y
    addQuad(out, p[3], p[7], p[6], p[2], color);  // +Y
    return out;
}

std::vector<Triangle> buildSphere(float radius, int segments, const Eigen::Vector3f& color) {
    int rings = std::max(3, segments);
    int sectors = std::max(3, segments * 2);
    std::vector<Triangle> out;
    auto at = [&](int i, int j) -> Eigen::Vector3f {
        float v = kPi * (float)i / rings;          // 0..pi (pole to pole)
        float u = 2.0f * kPi * (float)j / sectors; // 0..2pi
        return {radius * std::sin(v) * std::cos(u), radius * std::cos(v),
                radius * std::sin(v) * std::sin(u)};
    };
    for (int i = 0; i < rings; ++i)
        for (int j = 0; j < sectors; ++j) {
            Eigen::Vector3f a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
            Eigen::Vector3f na = a.normalized(), nb = b.normalized(), nc = c.normalized(),
                            nd = d.normalized();
            addSmooth(out, a, b, c, na, nb, nc, color);
            addSmooth(out, a, c, d, na, nc, nd, color);
        }
    return out;
}

std::vector<Triangle> buildCylinder(float radius, float height, int segments,
                                    const Eigen::Vector3f& color) {
    int n = std::max(3, segments);
    float h = height * 0.5f;
    std::vector<Triangle> out;
    for (int j = 0; j < n; ++j) {
        float u0 = 2.0f * kPi * j / n, u1 = 2.0f * kPi * (j + 1) / n;
        Eigen::Vector3f d0(std::cos(u0), 0, std::sin(u0)), d1(std::cos(u1), 0, std::sin(u1));
        Eigen::Vector3f b0 = radius * d0 - Eigen::Vector3f(0, h, 0),
                        b1 = radius * d1 - Eigen::Vector3f(0, h, 0);
        Eigen::Vector3f t0 = radius * d0 + Eigen::Vector3f(0, h, 0),
                        t1 = radius * d1 + Eigen::Vector3f(0, h, 0);
        addSmooth(out, b0, b1, t1, d0, d1, d1, color);  // side (smooth radial normals)
        addSmooth(out, b0, t1, t0, d0, d1, d0, color);
        addFlat(out, {0, -h, 0}, b1, b0, color);        // bottom cap
        addFlat(out, {0, h, 0}, t0, t1, color);         // top cap
    }
    return out;
}

std::vector<Triangle> buildCone(float radius, float height, int segments,
                                const Eigen::Vector3f& color) {
    int n = std::max(3, segments);
    float h = height * 0.5f;
    Eigen::Vector3f apex(0, h, 0);
    std::vector<Triangle> out;
    for (int j = 0; j < n; ++j) {
        float u0 = 2.0f * kPi * j / n, u1 = 2.0f * kPi * (j + 1) / n;
        Eigen::Vector3f b0 = Eigen::Vector3f(radius * std::cos(u0), -h, radius * std::sin(u0));
        Eigen::Vector3f b1 = Eigen::Vector3f(radius * std::cos(u1), -h, radius * std::sin(u1));
        addFlat(out, b0, b1, apex, color);        // side
        addFlat(out, {0, -h, 0}, b1, b0, color);  // base
    }
    return out;
}

std::vector<Triangle> buildTorus(float majorRadius, float minorRadius, int segMajor,
                                 int segMinor, const Eigen::Vector3f& color) {
    int nu = std::max(3, segMajor), nv = std::max(3, segMinor);
    std::vector<Triangle> out;
    auto at = [&](int i, int j, Eigen::Vector3f& pos, Eigen::Vector3f& nrm) {
        float u = 2.0f * kPi * i / nu, v = 2.0f * kPi * j / nv;
        Eigen::Vector3f center(majorRadius * std::cos(u), 0, majorRadius * std::sin(u));
        Eigen::Vector3f n(std::cos(v) * std::cos(u), std::sin(v), std::cos(v) * std::sin(u));
        nrm = n.normalized();
        pos = center + minorRadius * nrm;
    };
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j) {
            Eigen::Vector3f a, b, c, d, na, nb, nc, nd;
            at(i, j, a, na); at(i + 1, j, b, nb); at(i + 1, j + 1, c, nc); at(i, j + 1, d, nd);
            addSmooth(out, a, b, c, na, nb, nc, color);
            addSmooth(out, a, c, d, na, nc, nd, color);
        }
    return out;
}

std::vector<Triangle> buildDisk(float radius, int segments, const Eigen::Vector3f& color) {
    int n = std::max(3, segments);
    std::vector<Triangle> out;
    for (int j = 0; j < n; ++j) {
        float u0 = 2.0f * kPi * j / n, u1 = 2.0f * kPi * (j + 1) / n;
        Eigen::Vector3f p0(radius * std::cos(u0), 0, radius * std::sin(u0));
        Eigen::Vector3f p1(radius * std::cos(u1), 0, radius * std::sin(u1));
        addFlat(out, {0, 0, 0}, p0, p1, color);
    }
    return out;
}

std::vector<Triangle> buildCapsule(float radius, float cylinderHeight, int segments,
                                   const Eigen::Vector3f& color) {
    int n = std::max(3, segments);
    int rings = std::max(2, segments / 2);
    float h = cylinderHeight * 0.5f;
    std::vector<Triangle> out;
    // Cylindrical body.
    for (int j = 0; j < n; ++j) {
        float u0 = 2.0f * kPi * j / n, u1 = 2.0f * kPi * (j + 1) / n;
        Eigen::Vector3f d0(std::cos(u0), 0, std::sin(u0)), d1(std::cos(u1), 0, std::sin(u1));
        Eigen::Vector3f b0 = radius * d0 - Eigen::Vector3f(0, h, 0),
                        b1 = radius * d1 - Eigen::Vector3f(0, h, 0);
        Eigen::Vector3f t0 = radius * d0 + Eigen::Vector3f(0, h, 0),
                        t1 = radius * d1 + Eigen::Vector3f(0, h, 0);
        addSmooth(out, b0, b1, t1, d0, d1, d1, color);
        addSmooth(out, b0, t1, t0, d0, d1, d0, color);
    }
    // Hemispherical caps (top: +half, offset +h; bottom: -half, offset -h).
    auto cap = [&](float sign, float yOff) {
        for (int i = 0; i < rings; ++i)
            for (int j = 0; j < n; ++j) {
                auto sph = [&](int ii, int jj) -> Eigen::Vector3f {
                    float v = 0.5f * kPi * (float)ii / rings;  // 0..pi/2
                    float u = 2.0f * kPi * (float)jj / n;
                    return {radius * std::cos(v) * std::cos(u), sign * radius * std::sin(v),
                            radius * std::cos(v) * std::sin(u)};
                };
                Eigen::Vector3f off(0, yOff, 0);
                Eigen::Vector3f a = sph(i, j) + off, b = sph(i + 1, j) + off,
                                c = sph(i + 1, j + 1) + off, d = sph(i, j + 1) + off;
                Eigen::Vector3f na = (a - off).normalized(), nb = (b - off).normalized(),
                                nc = (c - off).normalized(), nd = (d - off).normalized();
                addSmooth(out, a, b, c, na, nb, nc, color);
                addSmooth(out, a, c, d, na, nc, nd, color);
            }
    };
    cap(1.0f, h);
    cap(-1.0f, -h);
    return out;
}

std::vector<Triangle> buildArrow(float length, float radius, int segments,
                                 const Eigen::Vector3f& color) {
    // Shaft (cylinder) along +Y for the lower 70%, head (cone) for the top 30%.
    float shaftLen = length * 0.7f, headLen = length * 0.3f;
    std::vector<Triangle> out = buildCylinder(radius, shaftLen, segments, color);
    for (auto& t : out)
        for (auto& v : t.v) v.y() += shaftLen * 0.5f;  // base at y=0
    std::vector<Triangle> head = buildCone(radius * 2.0f, headLen, segments, color);
    for (auto& t : head)
        for (auto& v : t.v) v.y() += shaftLen + headLen * 0.5f;
    out.insert(out.end(), head.begin(), head.end());
    return out;
}

} // namespace orange::geometry
