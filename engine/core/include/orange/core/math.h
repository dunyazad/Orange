#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

// Linear algebra for the renderer, backed by Eigen. Types are plain Eigen
// types (column-major, the OpenGL convention) so they interop with the render
// ABI via .data() (a column-major float[16]/float[3]). The free functions below
// keep the graphics-specific builders (perspective/ortho/lookAt) and a handful
// of thin wrappers so call sites read the same as before.
//
// NOTE: the project builds with EIGEN_MAX_ALIGN_BYTES=0 (see CMake) so these
// types carry no over-alignment and are safe to store in EnTT component pools.

namespace orange::math {

using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat4 = Eigen::Matrix4f;
using Quat = Eigen::Quaternionf;

// --- Vector helpers --------------------------------------------------------
inline float dot(const Vec3& a, const Vec3& b) { return a.dot(b); }
inline Vec3  cross(const Vec3& a, const Vec3& b) { return a.cross(b); }
inline Vec3  normalize(const Vec3& v) {
    float len = v.norm();
    return len > 1e-8f ? Vec3(v / len) : Vec3(0, 0, 0);
}

// --- Affine builders (column-major, OpenGL convention) ---------------------
inline Mat4 translate(const Vec3& t) {
    Mat4 m = Mat4::Identity();
    m.block<3, 1>(0, 3) = t;
    return m;
}

inline Mat4 scale(const Vec3& s) {
    Mat4 m = Mat4::Identity();
    m(0, 0) = s.x();
    m(1, 1) = s.y();
    m(2, 2) = s.z();
    return m;
}

inline Mat4 rotateX(float a) {
    Mat4 m = Mat4::Identity();
    m.block<3, 3>(0, 0) = Eigen::AngleAxisf(a, Vec3::UnitX()).toRotationMatrix();
    return m;
}
inline Mat4 rotateY(float a) {
    Mat4 m = Mat4::Identity();
    m.block<3, 3>(0, 0) = Eigen::AngleAxisf(a, Vec3::UnitY()).toRotationMatrix();
    return m;
}
inline Mat4 rotateZ(float a) {
    Mat4 m = Mat4::Identity();
    m.block<3, 3>(0, 0) = Eigen::AngleAxisf(a, Vec3::UnitZ()).toRotationMatrix();
    return m;
}

// Euler XYZ (radians) -> rotation matrix (Ry * Rx * Rz), matching the previous
// hand-rolled convention.
inline Mat4 rotateEuler(const Vec3& e) {
    return rotateY(e.y()) * rotateX(e.x()) * rotateZ(e.z());
}

// Right-handed perspective, NDC depth in [-1, 1] (OpenGL).
inline Mat4 perspective(float fovYRadians, float aspect, float zNear, float zFar) {
    Mat4 m = Mat4::Zero();
    float f = 1.0f / std::tan(fovYRadians * 0.5f);
    m(0, 0) = f / aspect;
    m(1, 1) = f;
    m(2, 2) = (zFar + zNear) / (zNear - zFar);
    m(3, 2) = -1.0f;
    m(2, 3) = (2.0f * zFar * zNear) / (zNear - zFar);
    return m;
}

// Orthographic projection (right-handed, NDC depth -1..1).
inline Mat4 ortho(float l, float r, float b, float t, float n, float f) {
    Mat4 m = Mat4::Zero();
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
inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
    Vec3 f = normalize(center - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 m = Mat4::Identity();
    m.block<1, 3>(0, 0) = s.transpose();
    m.block<1, 3>(1, 0) = u.transpose();
    m.block<1, 3>(2, 0) = (-f).transpose();
    m(0, 3) = -s.dot(eye);
    m(1, 3) = -u.dot(eye);
    m(2, 3) = f.dot(eye);
    return m;
}

// --- Quaternion helpers ----------------------------------------------------
inline Quat normalize(const Quat& q) { return q.normalized(); }
inline Quat conjugate(const Quat& q) { return q.conjugate(); }
inline float dot(const Quat& a, const Quat& b) { return a.dot(b); }

inline Quat quatAxisAngle(const Vec3& axis, float radians) {
    float len = axis.norm();
    if (len < 1e-8f) return Quat::Identity();
    return Quat(Eigen::AngleAxisf(radians, axis / len));
}

// Rotate a vector by a unit quaternion.
inline Vec3 rotate(const Quat& q, const Vec3& v) { return q * v; }

// Shortest-path spherical interpolation between two unit quaternions.
inline Quat slerp(const Quat& a, const Quat& b, float t) { return a.slerp(t, b); }

inline Mat4 toMat4(const Quat& q) {
    Mat4 m = Mat4::Identity();
    m.block<3, 3>(0, 0) = q.toRotationMatrix();
    return m;
}

// Extract a quaternion from the rotation part of a matrix.
inline Quat quatFromMat3(const Mat4& m) {
    return Quat(Eigen::Matrix3f(m.block<3, 3>(0, 0))).normalized();
}

// Build a unit quaternion whose local +Z axis aligns with `dir` (used to orient
// a camera so its eye sits along +dir from the target). Picks a stable up.
inline Quat quatLookZ(const Vec3& dir) {
    Vec3 z = normalize(dir);
    Vec3 upRef = std::fabs(z.dot(Vec3(0, 1, 0))) > 0.99f ? Vec3(0, 0, 1)
                                                         : Vec3(0, 1, 0);
    Vec3 x = normalize(cross(upRef, z));
    Vec3 y = cross(z, x);
    Eigen::Matrix3f R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return Quat(R);
}

} // namespace orange::math
