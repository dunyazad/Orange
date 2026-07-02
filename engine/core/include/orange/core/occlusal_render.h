#pragma once

// Stage 2 of the tooth-segmentation pipeline: a multi-channel orthographic
// "render" of the mesh along the occlusal normal. The mesh is projected onto a
// res x res grid in the occlusal plane; per cell the topmost (max-height) sample
// contributes to three channels that feed a downstream 2D segmenter:
//   - depth     : height along the occlusal normal (jet heatmap),
//   - normal    : surface normal, encoded n*0.5+0.5 -> RGB,
//   - curvature : PCA surface variation (jet heatmap) -- highlights the
//                 tooth/gum boundary and interproximal valleys.
// Each channel is an RGBA8 image (top row first) ready for createTexture.
//
// ASCII comments only (pipeline project rule).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace orange::geometry {

struct OcclusalChannels {
    int width = 0, height = 0;
    std::vector<uint8_t> depth;      // RGBA8
    std::vector<uint8_t> normal;     // RGBA8
    std::vector<uint8_t> curvature;  // RGBA8
    std::vector<uint8_t> segment;    // RGBA8: classical 2D watershed labels (per-tooth color)
    bool valid = false;

    // In-plane projection so callers can map a world point to image pixels:
    //   a = (p - origin).dot(u);  px = (a - aOff) / cell
    //   b = (p - origin).dot(v);  row = (height-1) - (b - bOff) / cell
    Eigen::Vector3f origin = Eigen::Vector3f::Zero(), u = Eigen::Vector3f::UnitX(),
                    v = Eigen::Vector3f::UnitY();
    float aOff = 0.0f, bOff = 0.0f, cell = 1.0f;
};

// Project `verts` (world space) along `planeNormal` (through `planePos`) onto a
// res x res grid and rasterize the three channels. Normals + curvature are
// estimated per vertex (kNN PCA). Empty cells are transparent.
OcclusalChannels renderOcclusalChannels(const std::vector<Eigen::Vector3f>& verts,
                                        const Eigen::Vector3f& planePos,
                                        const Eigen::Vector3f& planeNormal,
                                        int res = 256, float segProminence = 0.20f);

} // namespace orange::geometry
