#pragma once

// On-device MobileSAM (ONNX Runtime) 2D segmentation for pipeline stage 3b.
// Runs entirely locally -- the image never leaves the machine. Given the occlusal
// render image + a set of prompt points (the cusp tips, one+ per tooth), it
// encodes the image once and runs the mask decoder per point, then merges masks
// that overlap (same tooth) into per-tooth labels and returns a colored RGBA
// image. When ONNX Runtime was not built in, samAvailable() is false and
// samSegment() returns an invalid result.
//
// ASCII comments only (pipeline project rule).

#include <cstdint>
#include <utility>
#include <vector>

namespace orange::ml {

struct SamResult {
    int                  width = 0, height = 0;
    std::vector<uint8_t> rgba;        // RGBA8 labeled image (empty pixels transparent)
    int                  regions = 0;  // distinct tooth labels
    bool                 valid = false;
};

// True if ONNX Runtime + the MobileSAM models are compiled in.
bool samAvailable();

// `rgb` is width*height*3 bytes (RGB, row-major top row first). `points` are
// prompt locations in image pixels (x, y).
SamResult samSegment(const std::vector<uint8_t>& rgb, int width, int height,
                     const std::vector<std::pair<float, float>>& points);

} // namespace orange::ml
