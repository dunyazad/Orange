#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include <Eigen/Core>

// Ray + axis-aligned bounding box primitives with the usual intersection tests.
// Ported from Elements/Helium (TypeDefinitions.h), Eigen-backed, header-only.
// Orange builds picking rays inline today; these give reusable typed primitives
// for the geometry toolkit (sparse grids, mesh queries, debug draw).

namespace orange::geometry {

struct Ray {
    Eigen::Vector3f origin;
    Eigen::Vector3f direction;
    Eigen::Vector3f inverseDirection;

    Ray(const Eigen::Vector3f& o, const Eigen::Vector3f& d) : origin(o), direction(d) {
        const float eps = 1e-6f;
        for (int i = 0; i < 3; ++i) {
            inverseDirection[i] = (std::abs(d[i]) < eps)
                                      ? ((d[i] >= 0) ? 1e20f : -1e20f)
                                      : (1.0f / d[i]);
        }
    }

    // Nearest non-negative hit t against a sphere; false if no forward hit.
    bool intersectSphere(const Eigen::Vector3f& center, float radius, float& t) const {
        Eigen::Vector3f m = origin - center;
        float b = m.dot(direction);
        float c = m.dot(m) - radius * radius;
        if (c > 0.0f && b > 0.0f) return false;
        float discr = b * b - c;
        if (discr < 0.0f) return false;
        t = -b - std::sqrt(discr);
        if (t < 0.0f) t = -b + std::sqrt(discr);
        return t >= 0.0f;
    }

    // Plane given by normal . x + d = 0.
    bool intersectPlane(const Eigen::Vector3f& normal, float planeD, float& t) const {
        float denom = normal.dot(direction);
        if (std::abs(denom) < 1e-6f) return false;
        t = -(normal.dot(origin) + planeD) / denom;
        return t >= 0.0f;
    }

    // Moeller-Trumbore triangle intersection.
    bool intersectTriangle(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1,
                           const Eigen::Vector3f& v2, float& t) const {
        Eigen::Vector3f edge1 = v1 - v0;
        Eigen::Vector3f edge2 = v2 - v0;
        Eigen::Vector3f h = direction.cross(edge2);
        float a = edge1.dot(h);
        if (std::abs(a) < 1e-6f) return false;
        float f = 1.0f / a;
        Eigen::Vector3f s = origin - v0;
        float u = f * s.dot(h);
        if (u < 0.0f || u > 1.0f) return false;
        Eigen::Vector3f q = s.cross(edge1);
        float v = f * direction.dot(q);
        if (v < 0.0f || u + v > 1.0f) return false;
        t = f * edge2.dot(q);
        return t >= 0.0f;
    }
};

struct AABB {
    Eigen::Vector3f min = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
    Eigen::Vector3f max = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    bool intersects(const AABB& o) const {
        if (max.x() < o.min.x() || min.x() > o.max.x()) return false;
        if (max.y() < o.min.y() || min.y() > o.max.y()) return false;
        if (max.z() < o.min.z() || min.z() > o.max.z()) return false;
        return true;
    }

    bool contains(const Eigen::Vector3f& p) const {
        return p.x() >= min.x() && p.x() <= max.x() && p.y() >= min.y() && p.y() <= max.y() &&
               p.z() >= min.z() && p.z() <= max.z();
    }

    void expand(const Eigen::Vector3f& p) {
        min = min.cwiseMin(p);
        max = max.cwiseMax(p);
    }
    void expand(const AABB& o) {
        min = min.cwiseMin(o.min);
        max = max.cwiseMax(o.max);
    }

    // Slab test; returns the near/far parametric hit range along the ray.
    bool intersectRay(const Ray& ray, float& tNear, float& tFar) const {
        Eigen::Vector3f t0 = (min - ray.origin).cwiseProduct(ray.inverseDirection);
        Eigen::Vector3f t1 = (max - ray.origin).cwiseProduct(ray.inverseDirection);
        tNear = t0.cwiseMin(t1).maxCoeff();
        tFar  = t0.cwiseMax(t1).minCoeff();
        return tNear <= tFar && tFar >= 0.0f;
    }
};

// Per-axis percentile AABB ([pct, 1-pct]) so a few far outliers don't inflate the
// bounds. Falls back to the exact AABB for small sets. Used to keep spatial-viz
// cell sizes (and derived line thickness) tied to the model, not to stray points.
inline AABB robustBounds(const std::vector<Eigen::Vector3f>& pts, float pct = 0.01f) {
    AABB b;
    if (pts.empty()) return b;
    if (pts.size() < 50) {
        for (const auto& p : pts) b.expand(p);
        return b;
    }
    std::vector<float> v(pts.size());
    for (int a = 0; a < 3; ++a) {
        for (size_t i = 0; i < pts.size(); ++i) v[i] = pts[i][a];
        size_t lo = static_cast<size_t>(v.size() * pct);
        size_t hi = v.size() - 1 - lo;
        std::nth_element(v.begin(), v.begin() + lo, v.end()); b.min[a] = v[lo];
        std::nth_element(v.begin(), v.begin() + hi, v.end()); b.max[a] = v[hi];
    }
    return b;
}

} // namespace orange::geometry
