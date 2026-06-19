#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

// Linear algebra for the renderer, backed by Eigen. Call sites use the Eigen
// types directly (Eigen::Vector3f / Vector4f / Matrix4f / Quaternionf), which
// are column-major (the OpenGL convention) so they interop with the render ABI
// via .data() (a column-major float[16]/float[3]). The free functions below
// keep the graphics-specific builders (perspective/ortho/lookAt) and a handful
// of thin wrappers so call sites read clearly.
//
// NOTE: the project builds with EIGEN_MAX_ALIGN_BYTES=0 (see CMake) so these
// types carry no over-alignment and are safe to store in EnTT component pools.

namespace orange::math {

// --- Vector helpers --------------------------------------------------------
inline float dot(const Eigen::Vector3f& a, const Eigen::Vector3f& b) { return a.dot(b); }
inline Eigen::Vector3f cross(const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
    return a.cross(b);
}
inline Eigen::Vector3f normalize(const Eigen::Vector3f& v) {
    float len = v.norm();
    return len > 1e-8f ? Eigen::Vector3f(v / len) : Eigen::Vector3f(0, 0, 0);
}

// --- Affine builders (column-major, OpenGL convention) ---------------------
inline Eigen::Matrix4f translate(const Eigen::Vector3f& t) {
    return Eigen::Affine3f(Eigen::Translation3f(t)).matrix();
}

inline Eigen::Matrix4f scale(const Eigen::Vector3f& s) {
    return Eigen::Affine3f(Eigen::Scaling(s.x(), s.y(), s.z())).matrix();
}

inline Eigen::Matrix4f rotateX(float a) {
    return Eigen::Affine3f(Eigen::AngleAxisf(a, Eigen::Vector3f::UnitX())).matrix();
}
inline Eigen::Matrix4f rotateY(float a) {
    return Eigen::Affine3f(Eigen::AngleAxisf(a, Eigen::Vector3f::UnitY())).matrix();
}
inline Eigen::Matrix4f rotateZ(float a) {
    return Eigen::Affine3f(Eigen::AngleAxisf(a, Eigen::Vector3f::UnitZ())).matrix();
}

// Euler XYZ (radians) -> rotation matrix (Ry * Rx * Rz), matching the previous
// hand-rolled convention.
inline Eigen::Matrix4f rotateEuler(const Eigen::Vector3f& e) {
    return Eigen::Affine3f(Eigen::AngleAxisf(e.y(), Eigen::Vector3f::UnitY()) *
                           Eigen::AngleAxisf(e.x(), Eigen::Vector3f::UnitX()) *
                           Eigen::AngleAxisf(e.z(), Eigen::Vector3f::UnitZ()))
        .matrix();
}

// Right-handed perspective, NDC depth in [-1, 1] (OpenGL).
inline Eigen::Matrix4f perspective(float fovYRadians, float aspect, float zNear, float zFar) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    float f = 1.0f / std::tan(fovYRadians * 0.5f);
    m(0, 0) = f / aspect;
    m(1, 1) = f;
    m(2, 2) = (zFar + zNear) / (zNear - zFar);
    m(3, 2) = -1.0f;
    m(2, 3) = (2.0f * zFar * zNear) / (zNear - zFar);
    return m;
}

// Orthographic projection (right-handed, NDC depth -1..1).
inline Eigen::Matrix4f ortho(float l, float r, float b, float t, float n, float f) {
    Eigen::Matrix4f m = Eigen::Matrix4f::Zero();
    m(0, 0) = 2.0f / (r - l);
    m(1, 1) = 2.0f / (t - b);
    m(2, 2) = -2.0f / (f - n);
    m(0, 3) = -(r + l) / (r - l);
    m(1, 3) = -(t + b) / (t - b);
    m(2, 3) = -(f + n) / (f - n);
    m(3, 3) = 1.0f;
    return m;
}

// Right-handed look-at.
inline Eigen::Matrix4f lookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& center,
                              const Eigen::Vector3f& up) {
    Eigen::Vector3f f = normalize(center - eye);
    Eigen::Vector3f s = normalize(cross(f, up));
    Eigen::Vector3f u = cross(s, f);
    Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
    m.block<1, 3>(0, 0) = s.transpose();
    m.block<1, 3>(1, 0) = u.transpose();
    m.block<1, 3>(2, 0) = (-f).transpose();
    m(0, 3) = -s.dot(eye);
    m(1, 3) = -u.dot(eye);
    m(2, 3) = f.dot(eye);
    return m;
}

// --- Quaternion helpers ----------------------------------------------------
inline Eigen::Quaternionf normalize(const Eigen::Quaternionf& q) { return q.normalized(); }
inline Eigen::Quaternionf conjugate(const Eigen::Quaternionf& q) { return q.conjugate(); }
inline float dot(const Eigen::Quaternionf& a, const Eigen::Quaternionf& b) { return a.dot(b); }

inline Eigen::Quaternionf quatAxisAngle(const Eigen::Vector3f& axis, float radians) {
    float len = axis.norm();
    if (len < 1e-8f) return Eigen::Quaternionf::Identity();
    return Eigen::Quaternionf(Eigen::AngleAxisf(radians, axis / len));
}

// Rotate a vector by a unit quaternion.
inline Eigen::Vector3f rotate(const Eigen::Quaternionf& q, const Eigen::Vector3f& v) {
    return q * v;
}

// Shortest-path spherical interpolation between two unit quaternions.
inline Eigen::Quaternionf slerp(const Eigen::Quaternionf& a, const Eigen::Quaternionf& b, float t) {
    return a.slerp(t, b);
}

inline Eigen::Matrix4f toMat4(const Eigen::Quaternionf& q) {
    return Eigen::Affine3f(q).matrix();
}

// Extract a quaternion from the rotation part of a matrix.
inline Eigen::Quaternionf quatFromMat3(const Eigen::Matrix4f& m) {
    return Eigen::Quaternionf(Eigen::Matrix3f(m.block<3, 3>(0, 0))).normalized();
}

// Build a unit quaternion whose local +Z axis aligns with `dir` (used to orient
// a camera so its eye sits along +dir from the target). Picks a stable up.
inline Eigen::Quaternionf quatLookZ(const Eigen::Vector3f& dir) {
    Eigen::Vector3f z = normalize(dir);
    Eigen::Vector3f upRef = std::fabs(z.dot(Eigen::Vector3f(0, 1, 0))) > 0.99f
                                ? Eigen::Vector3f(0, 0, 1)
                                : Eigen::Vector3f(0, 1, 0);
    Eigen::Vector3f x = normalize(cross(upRef, z));
    Eigen::Vector3f y = cross(z, x);
    Eigen::Matrix3f R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return Eigen::Quaternionf(R);
}

} // namespace orange::math
