#pragma once

#include <cmath>

// Tiny, self-contained linear algebra (column-major, OpenGL convention).
// Just enough for a renderer: no external math dependency.

namespace orange::math {

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3  cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 normalize(Vec3 v) {
    float len = std::sqrt(dot(v, v));
    if (len <= 1e-8f) return {0, 0, 0};
    return v * (1.0f / len);
}

// 4x4 matrix, column-major: element (col,row) lives at m[col*4 + row].
struct Mat4 {
    float m[16] = {0};

    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}

inline Mat4 translate(Vec3 t) {
    Mat4 r = Mat4::identity();
    r.m[12] = t.x;
    r.m[13] = t.y;
    r.m[14] = t.z;
    return r;
}

inline Mat4 scale(Vec3 s) {
    Mat4 r = Mat4::identity();
    r.m[0]  = s.x;
    r.m[5]  = s.y;
    r.m[10] = s.z;
    return r;
}

inline Mat4 rotateX(float a) {
    Mat4 r = Mat4::identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[5] = c;  r.m[6] = s;
    r.m[9] = -s; r.m[10] = c;
    return r;
}

inline Mat4 rotateY(float a) {
    Mat4 r = Mat4::identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c;  r.m[2] = -s;
    r.m[8] = s;  r.m[10] = c;
    return r;
}

inline Mat4 rotateZ(float a) {
    Mat4 r = Mat4::identity();
    float c = std::cos(a), s = std::sin(a);
    r.m[0] = c;  r.m[1] = s;
    r.m[4] = -s; r.m[5] = c;
    return r;
}

// Euler XYZ (radians) -> rotation matrix (Ry * Rx * Rz).
inline Mat4 rotateEuler(Vec3 e) {
    return rotateY(e.y) * rotateX(e.x) * rotateZ(e.z);
}

// Right-handed perspective, NDC depth in [-1, 1] (OpenGL).
inline Mat4 perspective(float fovYRadians, float aspect, float zNear, float zFar) {
    Mat4 r;  // zero-initialized
    float f = 1.0f / std::tan(fovYRadians * 0.5f);
    r.m[0]  = f / aspect;
    r.m[5]  = f;
    r.m[10] = (zFar + zNear) / (zNear - zFar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
    return r;
}

// --- Quaternion ------------------------------------------------------------
// Orientation as a unit quaternion. Used wherever rotation must accumulate
// without gimbal lock (camera trackball, rigid-body spin). Layout: (x,y,z) is
// the vector part, w the scalar part.
struct Quat {
    float x = 0, y = 0, z = 0, w = 1;  // identity by default

    static Quat identity() { return {0, 0, 0, 1}; }
};

// Hamilton product: applying `a` after `b` (a * b).
inline Quat operator*(const Quat& a, const Quat& b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}

inline Quat normalize(const Quat& q) {
    float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (len <= 1e-8f) return Quat::identity();
    float inv = 1.0f / len;
    return {q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

inline Quat quatAxisAngle(Vec3 axis, float radians) {
    Vec3 a = normalize(axis);
    if (a.x == 0 && a.y == 0 && a.z == 0) return Quat::identity();
    float s = std::sin(radians * 0.5f);
    return {a.x * s, a.y * s, a.z * s, std::cos(radians * 0.5f)};
}

// Rotate a vector by a unit quaternion: v + 2w(q x v) + 2 q x (q x v).
inline Vec3 rotate(const Quat& q, Vec3 v) {
    Vec3 u{q.x, q.y, q.z};
    Vec3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}

inline Quat conjugate(const Quat& q) { return {-q.x, -q.y, -q.z, q.w}; }

inline float dot(const Quat& a, const Quat& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// Shortest-path spherical interpolation between two unit quaternions.
inline Quat slerp(Quat a, Quat b, float t) {
    float d = dot(a, b);
    if (d < 0.0f) {  // take the shorter arc
        b = {-b.x, -b.y, -b.z, -b.w};
        d = -d;
    }
    if (d > 0.9995f) {  // nearly parallel -> linear, then renormalize
        return normalize(Quat{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                              a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t});
    }
    float theta = std::acos(d);
    float s = std::sin(theta);
    float wa = std::sin((1.0f - t) * theta) / s;
    float wb = std::sin(t * theta) / s;
    return {a.x * wa + b.x * wb, a.y * wa + b.y * wb, a.z * wa + b.z * wb,
            a.w * wa + b.w * wb};
}

inline Mat4 toMat4(const Quat& q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    Mat4 r = Mat4::identity();
    r.m[0]  = 1 - 2 * (y * y + z * z);
    r.m[1]  = 2 * (x * y + w * z);
    r.m[2]  = 2 * (x * z - w * y);
    r.m[4]  = 2 * (x * y - w * z);
    r.m[5]  = 1 - 2 * (x * x + z * z);
    r.m[6]  = 2 * (y * z + w * x);
    r.m[8]  = 2 * (x * z + w * y);
    r.m[9]  = 2 * (y * z - w * x);
    r.m[10] = 1 - 2 * (x * x + y * y);
    return r;
}

// Build a unit quaternion whose local +Z axis aligns with `dir` (used to orient
// a camera so its eye sits along +dir from the target). Picks a stable up.
inline Quat quatFromMat3(const Mat4& m);  // fwd decl
inline Quat quatLookZ(Vec3 dir) {
    Vec3 z = normalize(dir);
    Vec3 upRef = std::fabs(dot(z, Vec3{0, 1, 0})) > 0.99f ? Vec3{0, 0, 1}
                                                          : Vec3{0, 1, 0};
    Vec3 x = normalize(cross(upRef, z));
    Vec3 y = cross(z, x);
    Mat4 R = Mat4::identity();
    R.m[0] = x.x; R.m[1] = x.y; R.m[2] = x.z;
    R.m[4] = y.x; R.m[5] = y.y; R.m[6] = y.z;
    R.m[8] = z.x; R.m[9] = z.y; R.m[10] = z.z;
    return quatFromMat3(R);
}

// Extract a quaternion from the rotation part of a matrix (Shepperd's method).
inline Quat quatFromMat3(const Mat4& m) {
    float tr = m.m[0] + m.m[5] + m.m[10];
    Quat q;
    if (tr > 0.0f) {
        float s = std::sqrt(tr + 1.0f) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m.m[6] - m.m[9]) / s;
        q.y = (m.m[8] - m.m[2]) / s;
        q.z = (m.m[1] - m.m[4]) / s;
    } else if (m.m[0] > m.m[5] && m.m[0] > m.m[10]) {
        float s = std::sqrt(1.0f + m.m[0] - m.m[5] - m.m[10]) * 2.0f;
        q.w = (m.m[6] - m.m[9]) / s;
        q.x = 0.25f * s;
        q.y = (m.m[4] + m.m[1]) / s;
        q.z = (m.m[8] + m.m[2]) / s;
    } else if (m.m[5] > m.m[10]) {
        float s = std::sqrt(1.0f + m.m[5] - m.m[0] - m.m[10]) * 2.0f;
        q.w = (m.m[8] - m.m[2]) / s;
        q.x = (m.m[4] + m.m[1]) / s;
        q.y = 0.25f * s;
        q.z = (m.m[9] + m.m[6]) / s;
    } else {
        float s = std::sqrt(1.0f + m.m[10] - m.m[0] - m.m[5]) * 2.0f;
        q.w = (m.m[1] - m.m[4]) / s;
        q.x = (m.m[8] + m.m[2]) / s;
        q.y = (m.m[9] + m.m[6]) / s;
        q.z = 0.25f * s;
    }
    return normalize(q);
}

// Orthographic projection (right-handed, NDC depth -1..1).
inline Mat4 ortho(float l, float r, float b, float t, float n, float f) {
    Mat4 m;  // zero
    m.m[0]  = 2.0f / (r - l);
    m.m[5]  = 2.0f / (t - b);
    m.m[10] = -2.0f / (f - n);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = -(t + b) / (t - b);
    m.m[14] = -(f + n) / (f - n);
    m.m[15] = 1.0f;
    return m;
}

// Right-handed look-at.
inline Mat4 lookAt(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0] = s.x; r.m[4] = s.y; r.m[8]  = s.z;
    r.m[1] = u.x; r.m[5] = u.y; r.m[9]  = u.z;
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

} // namespace orange::math
