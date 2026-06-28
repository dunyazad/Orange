// MobileSAM ONNX inference for on-device 2D tooth segmentation. See
// sam_segment.h. ASCII comments only (pipeline project rule).

#include "orange/core/sam_segment.h"

#include <cmath>
#include <cstdint>

namespace orange::ml {

#ifndef ORANGE_ENABLE_ONNX

bool samAvailable() { return false; }
SamResult samSegment(const std::vector<uint8_t>&, int, int,
                     const std::vector<std::pair<float, float>>&) {
    return {};
}

#else

} // namespace orange::ml

#include <array>
#include <mutex>
#include <string>

#include <onnxruntime_cxx_api.h>

namespace orange::ml {

bool samAvailable() { return true; }

namespace {

// Lazily-created, process-wide ORT env + the two sessions (encoder/decoder).
struct Sam {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "orange-sam"};
    Ort::SessionOptions opts;
    Ort::Session encoder{nullptr};
    Ort::Session decoder{nullptr};
    bool ready = false;

    Sam() {
        try {
            opts.SetIntraOpNumThreads(4);
            opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
            const std::string dir = ORANGE_ONNX_MODEL_DIR;
            std::wstring enc(dir.begin(), dir.end());
            enc += L"/encoder.onnx";
            std::wstring dec(dir.begin(), dir.end());
            dec += L"/decoder_multi.onnx";  // 4 masks/point -> pick the tooth-scale one
            encoder = Ort::Session(env, enc.c_str(), opts);
            decoder = Ort::Session(env, dec.c_str(), opts);
            ready = true;
        } catch (const Ort::Exception&) {
            ready = false;
        }
    }
};

Sam& sam() {
    static Sam s;
    return s;
}

// Distinct colour per integer label (hue wheel).
void labelColor(int lbl, uint8_t& r, uint8_t& g, uint8_t& b) {
    uint32_t h = (uint32_t)(lbl + 1) * 2654435761u;
    float hue = (float)(h & 0xFFFF) / 65535.0f * 6.0f;
    float x = 1.0f - std::fabs(std::fmod(hue, 2.0f) - 1.0f);
    float rf, gf, bf;
    if (hue < 1) { rf = 1; gf = x; bf = 0; }
    else if (hue < 2) { rf = x; gf = 1; bf = 0; }
    else if (hue < 3) { rf = 0; gf = 1; bf = x; }
    else if (hue < 4) { rf = 0; gf = x; bf = 1; }
    else if (hue < 5) { rf = x; gf = 0; bf = 1; }
    else { rf = 1; gf = 0; bf = x; }
    auto cv = [](float v) { return (uint8_t)((v * 0.6f + 0.25f) * 255.0f); };
    r = cv(rf); g = cv(gf); b = cv(bf);
}

} // namespace

SamResult samSegment(const std::vector<uint8_t>& rgb, int width, int height,
                     const std::vector<std::pair<float, float>>& points) {
    SamResult out;
    if (width <= 0 || height <= 0 || (int)rgb.size() < width * height * 3 || points.empty())
        return out;
    Sam& s = sam();
    if (!s.ready) return out;

    const size_t npx = (size_t)width * height;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    try {
        // --- Encoder: input_image [H,W,3] float (0..255; model normalizes) -----
        std::vector<float> img(npx * 3);
        for (size_t i = 0; i < npx * 3; ++i) img[i] = (float)rgb[i];
        std::array<int64_t, 3> eShape{height, width, 3};
        Ort::Value eIn = Ort::Value::CreateTensor<float>(mem, img.data(), img.size(),
                                                         eShape.data(), eShape.size());
        const char* eInNames[] = {"input_image"};
        const char* eOutNames[] = {"image_embeddings"};
        auto emb = s.encoder.Run(Ort::RunOptions{nullptr}, eInNames, &eIn, 1, eOutNames, 1);
        float* embData = emb[0].GetTensorMutableData<float>();
        const size_t embCount = 1 * 256 * 64 * 64;

        // --- Decoder per prompt point -> binary mask ---------------------------
        std::vector<int> label(npx, -1);  // per-pixel tooth label (-1 = background)
        std::vector<std::vector<uint8_t>> masks;  // accepted masks (per label)
        std::vector<int> area;
        int nextLabel = 0;

        std::array<int64_t, 4> embShape{1, 256, 64, 64};
        std::array<int64_t, 3> pcShape{1, 1, 2};
        std::array<int64_t, 2> plShape{1, 1};
        std::array<int64_t, 4> miShape{1, 1, 256, 256};
        std::array<int64_t, 1> hmShape{1};
        std::array<int64_t, 1> osShape{2};
        std::vector<float> maskInput(256 * 256, 0.0f);
        float hasMask = 0.0f;
        std::array<float, 2> origSize{(float)height, (float)width};

        const char* dInNames[] = {"image_embeddings", "point_coords", "point_labels",
                                  "mask_input", "has_mask_input", "orig_im_size"};
        const char* dOutNames[] = {"masks", "iou_predictions", "low_res_masks"};

        for (const auto& pt : points) {
            std::array<float, 2> pc{pt.first, pt.second};
            std::array<float, 1> pl{1.0f};  // foreground point

            std::array<Ort::Value, 6> dIn{
                Ort::Value::CreateTensor<float>(mem, embData, embCount, embShape.data(), embShape.size()),
                Ort::Value::CreateTensor<float>(mem, pc.data(), pc.size(), pcShape.data(), pcShape.size()),
                Ort::Value::CreateTensor<float>(mem, pl.data(), pl.size(), plShape.data(), plShape.size()),
                Ort::Value::CreateTensor<float>(mem, maskInput.data(), maskInput.size(), miShape.data(), miShape.size()),
                Ort::Value::CreateTensor<float>(mem, &hasMask, 1, hmShape.data(), hmShape.size()),
                Ort::Value::CreateTensor<float>(mem, origSize.data(), origSize.size(), osShape.data(), osShape.size()),
            };
            auto res = s.decoder.Run(Ort::RunOptions{nullptr}, dInNames, dIn.data(), dIn.size(),
                                     dOutNames, 3);

            // masks: [1, nMask, H, W] logits. Pick the smallest-area mask that is
            // a plausible tooth (good IoU, area in range) -- a single point grabs
            // the whole arch at the coarse scale, so the finer scales isolate the
            // tooth.
            auto shp = res[0].GetTensorTypeAndShapeInfo().GetShape();
            if (shp.size() != 4) continue;
            int nMask = (int)shp[1], mh = (int)shp[2], mw = (int)shp[3];
            if (mh != height || mw != width) continue;
            const float* m = res[0].GetTensorData<float>();
            const float* iou = res[1].GetTensorData<float>();

            const int areaMin = (int)npx / 300, areaMax = (int)(npx * 0.20f);
            int chosen = -1, chosenArea = 1 << 30;
            for (int km = 0; km < nMask; ++km) {
                const float* mk = m + (size_t)km * npx;
                int a = 0;
                for (size_t i = 0; i < npx; ++i) if (mk[i] > 0.0f) ++a;
                if (a < areaMin || a > areaMax) continue;   // too small/large for a tooth
                if (iou[km] < 0.5f) continue;
                if (a < chosenArea) { chosenArea = a; chosen = km; }
            }
            if (chosen < 0) continue;
            const float* mk = m + (size_t)chosen * npx;
            std::vector<uint8_t> bin(npx, 0);
            int a = 0;
            for (size_t i = 0; i < npx; ++i)
                if (mk[i] > 0.0f) { bin[i] = 1; ++a; }

            // Merge into an existing tooth if it overlaps a lot (same tooth, other cusp).
            int best = -1;
            float bestIoU = 0.0f;
            for (size_t k = 0; k < masks.size(); ++k) {
                int inter = 0;
                for (size_t i = 0; i < npx; ++i) inter += (bin[i] & masks[k][i]);
                float iou = (float)inter / (float)(a + area[k] - inter);
                if (iou > bestIoU) { bestIoU = iou; best = (int)k; }
            }
            int lbl;
            if (best >= 0 && bestIoU > 0.45f) {
                lbl = best;
                for (size_t i = 0; i < npx; ++i)
                    if (bin[i]) { masks[best][i] = 1; }
                area[best] = 0;
                for (size_t i = 0; i < npx; ++i) area[best] += masks[best][i];
            } else {
                lbl = nextLabel++;
                masks.push_back(bin);
                area.push_back(a);
            }
            for (size_t i = 0; i < npx; ++i)
                if (bin[i]) label[i] = lbl;
        }

        out.width = width; out.height = height;
        out.rgba.assign(npx * 4, 0);
        for (size_t i = 0; i < npx; ++i) {
            if (label[i] < 0) continue;
            uint8_t r, g, b;
            labelColor(label[i], r, g, b);
            out.rgba[i * 4 + 0] = r;
            out.rgba[i * 4 + 1] = g;
            out.rgba[i * 4 + 2] = b;
            out.rgba[i * 4 + 3] = 255;
        }
        out.regions = nextLabel;
        out.valid = true;
    } catch (const Ort::Exception&) {
        return {};
    }
    return out;
}

#endif  // ORANGE_ENABLE_ONNX

} // namespace orange::ml
