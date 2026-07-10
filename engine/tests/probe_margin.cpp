// Manual Recolor-mode probe on a real point cloud. Loads a PLY, runs the given
// Recolor mode (default "Margin Bump: Detect") with the given params, reports
// the per-color counts, and (optionally) writes a recolored PLY for visual
// inspection with `appOrange out.ply --shot`.
// Usage: orange_margin_probe in.ply [out.ply] [--mode "Bump: Detect"] [p0 p1 ...]

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "orange/core/debug_draw.h"
#include "orange/core/modes.h"
#include "orange/core/serialization.h"

using Eigen::Vector3f;
using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf(
            "usage: orange_margin_probe in.ply [out.ply] [--mode \"Bump: Detect\"] [p0 p1 ...]\n");
        return 2;
    }
    std::string inPath = argv[1];
    std::string outPath;
    std::string modeName = "Margin Bump: Detect";
    std::vector<float> params;
    bool doFit = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) modeName = argv[++i];
        else if (std::strcmp(argv[i], "--fit") == 0) doFit = true;
        else if (outPath.empty() && !std::isdigit((unsigned char)argv[i][0]) &&
                 argv[i][0] != '.' && argv[i][0] != '-')  // negative params are numbers too
            outPath = argv[i];
        else params.push_back((float)std::atof(argv[i]));
    }

    orange::io::PLYFormat ply;
    auto l0 = Clock::now();
    if (!ply.Deserialize(inPath)) { std::printf("FAILED to load %s\n", inPath.c_str()); return 2; }
    auto l1 = Clock::now();
    const std::vector<Vector3f>& P = ply.GetPoints();
    std::printf("loaded %zu points in %.0f ms\n", P.size(),
                std::chrono::duration<double, std::milli>(l1 - l0).count());
    if (P.size() < 32) { std::printf("too few points\n"); return 2; }

    Vector3f mn = P[0], mx = P[0];
    for (const auto& p : P) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    float diag = (mx - mn).norm();
    std::printf("bbox diag %.3f\n", diag);

    orange::modes::ModeInput mi;
    mi.points = P;
    if (ply.GetNormals().size() == P.size()) {
        mi.normals = ply.GetNormals();
        std::printf("using %zu file normals\n", mi.normals.size());
    } else if (!ply.GetTriangleIndices().empty()) {
        // Area-weighted vertex normals from the triangle winding -- oriented,
        // same fallback processingModeSystem uses for a normal-less mesh.
        mi.normals.assign(P.size(), Vector3f::Zero());
        for (const auto& f : ply.GetTriangleIndices()) {
            Vector3f fn = (P[f.y()] - P[f.x()]).cross(P[f.z()] - P[f.x()]);
            mi.normals[f.x()] += fn;
            mi.normals[f.y()] += fn;
            mi.normals[f.z()] += fn;
        }
        for (auto& n : mi.normals) {
            float l2 = n.squaredNorm();
            n = l2 > 1e-24f ? Vector3f(n / std::sqrt(l2)) : Vector3f::UnitY();
        }
        std::printf("computed %zu oriented normals from %zu faces\n", mi.normals.size(),
                    ply.GetTriangleIndices().size());
    }
    mi.params = params;
    int idx = orange::modes::modeIndexByName(modeName.c_str());
    if (idx < 0) { std::printf("mode '%s' not registered\n", modeName.c_str()); return 2; }
    std::printf("mode: %s\n", modeName.c_str());
    int np = orange::modes::modeParamCount(idx);
    for (int i = 0; i < np; ++i) {
        orange::modes::ModeParam mp = orange::modes::modeParam(idx, i);
        std::printf("  %-12s = %g%s\n", mp.name,
                    i < (int)mi.params.size() ? mi.params[i] : mp.defV,
                    i < (int)mi.params.size() ? "" : " (default)");
    }

    // --fit: apply the mode's fit action (runModeFit) and write the MOVED
    // cloud (original colors) instead of a recolored preview. Reports the
    // displacement stats so the smoothing strength is measurable.
    if (doFit) {
        std::vector<Vector3f> fitted;
        auto f0 = Clock::now();
        orange::modes::runModeFit(idx, mi, fitted, [](float f) {
            static int last = -1;
            int p = (int)(f * 100.0f);
            if (p / 10 != last) { last = p / 10; std::fprintf(stderr, " %d%%", p); }
        });
        auto f1 = Clock::now();
        std::fprintf(stderr, "\n");
        std::printf("fit ran in %.0f ms\n",
                    std::chrono::duration<double, std::milli>(f1 - f0).count());
        double sum = 0.0; float mxd = 0.0f;
        for (size_t i = 0; i < P.size(); ++i) {
            float d = (fitted[i] - P[i]).norm();
            sum += d;
            mxd = std::max(mxd, d);
        }
        std::printf("displacement mean %.5f  max %.5f  (diag %.3f)\n", sum / P.size(), mxd, diag);
        if (!outPath.empty()) {
            ply.GetPoints() = fitted;
            if (!ply.Serialize(outPath)) { std::printf("FAILED to write\n"); return 2; }
            std::printf("wrote %s\n", outPath.c_str());
        }
        return 0;
    }

    std::vector<Vector3f> colors;
    orange::debug::DebugDraw extras;
    auto t0 = Clock::now();
    orange::modes::runModeColors(idx, mi, colors, extras, [](float f) {
        static int last = -1;
        int p = (int)(f * 100.0f);
        if (p / 10 != last) { last = p / 10; std::fprintf(stderr, " %d%%", p); }
    });
    auto t1 = Clock::now();
    std::fprintf(stderr, "\n");
    std::printf("mode ran in %.0f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    // Classify by the known viz palettes: the bump/margin detect palette
    // (bright red / yellow / blue / keep-original) and the keep-drop filter
    // palette (kKeepGreen / kDropRed).
    const Vector3f red(1.0f, 0.05f, 0.05f), yellow(0.95f, 0.85f, 0.1f), blue(0.3f, 0.5f, 0.9f);
    const Vector3f keepGreen(0.1f, 0.9f, 0.2f), dropRed(0.5f, 0.12f, 0.12f);
    size_t nBump = 0, nBand = 0, nRoi = 0, nKeep = 0, nOther = 0;
    for (const auto& c : colors) {
        if (c.x() < 0.0f) ++nKeep;
        else if ((c - red).squaredNorm() < 1e-6f || (c - dropRed).squaredNorm() < 1e-6f) ++nBump;
        else if ((c - yellow).squaredNorm() < 1e-6f) ++nBand;
        else if ((c - blue).squaredNorm() < 1e-6f) ++nRoi;
        else if ((c - keepGreen).squaredNorm() < 1e-6f) ++nKeep;
        else ++nOther;
    }
    auto pct = [&](size_t n) { return 100.0 * (double)n / (double)P.size(); };
    std::printf("flagged (red) %8zu  (%.2f%%)\n", nBump, pct(nBump));
    std::printf("band (yellow) %8zu  (%.2f%%)\n", nBand, pct(nBand));
    std::printf("roi  (blue)   %8zu  (%.2f%%)\n", nRoi, pct(nRoi));
    std::printf("keep          %8zu  (%.2f%%)\n", nKeep, pct(nKeep));
    if (nOther) std::printf("other         %8zu  (%.2f%%)\n", nOther, pct(nOther));

    if (!outPath.empty()) {
        auto& outColors = ply.GetColors();
        outColors.resize(P.size(), Eigen::Vector4f(0.75f, 0.75f, 0.75f, 1.0f));
        size_t n = std::min(colors.size(), P.size());
        for (size_t i = 0; i < n; ++i)
            if (colors[i].x() >= 0.0f)
                outColors[i] = Eigen::Vector4f(colors[i].x(), colors[i].y(), colors[i].z(), 1.0f);
        if (!ply.Serialize(outPath)) { std::printf("FAILED to write %s\n", outPath.c_str()); return 2; }
        std::printf("wrote %s\n", outPath.c_str());
    }
    return 0;
}
