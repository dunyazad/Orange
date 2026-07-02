#pragma once

// Pipeline stage 5 (direct 3D): per-tooth segmentation on the mesh itself, no 2D
// projection. A watershed of the occlusal-height field over the mesh graph: every
// vertex flows uphill to a local maximum (a cusp), and basins are merged unless
// the saddle between them is a deep interproximal valley -- so each surviving
// basin is one tooth. The arch curvature is removed first (high-pass) so absolute
// height differences across the arch don't dominate; gingiva/base below a height
// gate are left unlabeled.
//
// ASCII comments only (pipeline project rule).

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace orange::geometry {

struct ToothSegParams {
    int   smoothIters = 4;     // light denoise of the height field
    int   baseIters   = 40;    // broad smooth = arch base (high-pass)
    float prominence  = 0.35f; // basin merge threshold (fraction of residual range)
    float heightGate  = 0.45f; // drop the lowest fraction (gingiva / base / palate)
    float curvThresh  = 0.0003f; // concavity (* bbox diag) above which a vertex is a
                                // boundary -- cuts at the gingival groove + interproximal
                                // valleys so a region stops at the tooth crown
};

struct ToothSegResult {
    std::vector<int> label;  // per input vertex; -1 = gingiva/base (unlabeled)
    int regions = 0;         // number of teeth
    bool valid = false;
};

// Segment teeth on the mesh (`vertices` + triangle `indices`). The result's
// `label` is indexed exactly like `vertices`.
ToothSegResult segmentTeeth(const std::vector<Eigen::Vector3f>& vertices,
                            const std::vector<uint32_t>& indices,
                            const ToothSegParams& params = {});

} // namespace orange::geometry
