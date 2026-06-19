#pragma once

#include <cmath>
#include <cstdint>

#include <Eigen/Core>

// Morton (Z-order curve) encoding for 3D voxel indices. Interleaves the bits of
// (x,y,z) into a single 64-bit key so spatially-near voxels get near keys --
// the basis for cache-coherent sparse voxel grids (see orange::geometry).
//
// Ported from Elements/Helium (Morton3D), CPU-only, Eigen-backed.

namespace orange::geometry {

using Vector3ui = Eigen::Matrix<unsigned int, 3, 1>;

class Morton3D {
public:
    // Interleave the low 21 bits of x,y,z into a 63-bit Morton key.
    static uint64_t encode(uint32_t x, uint32_t y, uint32_t z) {
        return part1By2(x) | (part1By2(y) << 1) | (part1By2(z) << 2);
    }
    static uint64_t encode(const Eigen::Vector3i& i) {
        return encode(static_cast<uint32_t>(i.x()), static_cast<uint32_t>(i.y()),
                      static_cast<uint32_t>(i.z()));
    }
    static uint64_t encode(const Vector3ui& i) { return encode(i.x(), i.y(), i.z()); }

    static void decode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) {
        x = compact1By2(code);
        y = compact1By2(code >> 1);
        z = compact1By2(code >> 2);
    }

    static Eigen::Vector3i keyToIndex(uint64_t code) {
        uint32_t x, y, z;
        decode(code, x, y, z);
        return Eigen::Vector3i(static_cast<int>(x), static_cast<int>(y), static_cast<int>(z));
    }
    static uint64_t indexToKey(const Eigen::Vector3i& i) { return encode(i); }
    static uint64_t indexToKey(const Vector3ui& i) { return encode(i); }

    // World position -> integer voxel index (floor) given a grid origin + size.
    static Vector3ui positionToIndex(const Eigen::Vector3f& p, const Eigen::Vector3f& origin,
                                     float voxelSize) {
        float inv = 1.0f / voxelSize;
        return Vector3ui(static_cast<unsigned>(std::floor((p.x() - origin.x()) * inv)),
                         static_cast<unsigned>(std::floor((p.y() - origin.y()) * inv)),
                         static_cast<unsigned>(std::floor((p.z() - origin.z()) * inv)));
    }
    static uint64_t encodeFromPosition(const Eigen::Vector3f& p, const Eigen::Vector3f& origin,
                                       float voxelSize) {
        return encode(positionToIndex(p, origin, voxelSize));
    }

    // Integer voxel index -> world position of the voxel center.
    static Eigen::Vector3f indexToPosition(const Vector3ui& i, const Eigen::Vector3f& origin,
                                           float voxelSize) {
        return origin + Eigen::Vector3f(static_cast<float>(i.x()) + 0.5f,
                                        static_cast<float>(i.y()) + 0.5f,
                                        static_cast<float>(i.z()) + 0.5f) *
                            voxelSize;
    }
    static Eigen::Vector3f decodeToPosition(uint64_t code, const Eigen::Vector3f& origin,
                                            float voxelSize) {
        uint32_t x, y, z;
        decode(code, x, y, z);
        return indexToPosition({x, y, z}, origin, voxelSize);
    }

private:
    // Insert two zero bits after each of the low 21 bits of x ("Part1By2").
    static uint64_t part1By2(uint32_t x) {
        uint64_t r = x;
        r &= 0x1fffff;
        r = (r | (r << 32)) & 0x1f00000000ffff;
        r = (r | (r << 16)) & 0x1f0000ff0000ff;
        r = (r | (r << 8)) & 0x100f00f00f00f00f;
        r = (r | (r << 4)) & 0x10c30c30c30c30c3;
        r = (r | (r << 2)) & 0x1249249249249249;
        return r;
    }
    // Inverse of part1By2: gather every third bit back into a contiguous value.
    static uint32_t compact1By2(uint64_t x) {
        x &= 0x1249249249249249;
        x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
        x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
        x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
        x = (x ^ (x >> 16)) & 0x1f00000000ffff;
        x = (x ^ (x >> 32)) & 0x1fffff;
        return static_cast<uint32_t>(x);
    }
};

} // namespace orange::geometry
