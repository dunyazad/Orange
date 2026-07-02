// 바이너리 STL을 읽어 findCusps를 직접 돌려 교두 개수를 확인하는 진단용 CLI
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include <Eigen/Core>

#include <algorithm>
#include <set>

#include "orange/core/half_edge.h"
#include "orange/core/occlusal_plane.h"
#include "orange/core/occlusal_render.h"
#include "orange/core/sam_segment.h"
#include "orange/core/tooth_segment.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: cusp_probe <file.stl>\n"); return 1; }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "open fail: %s\n", argv[1]); return 1; }

    char header[80];
    f.read(header, 80);
    uint32_t ntri = 0;
    f.read(reinterpret_cast<char*>(&ntri), 4);

    std::vector<Eigen::Vector3f> pos;  // triangle soup (3 verts per tri)
    std::vector<uint32_t> idx;
    pos.reserve((size_t)ntri * 3);
    idx.reserve((size_t)ntri * 3);
    for (uint32_t t = 0; t < ntri; ++t) {
        float buf[12];
        f.read(reinterpret_cast<char*>(buf), 48);  // normal(3) + 3 verts(9)
        uint16_t attr;
        f.read(reinterpret_cast<char*>(&attr), 2);
        if (!f) break;
        for (int k = 0; k < 3; ++k) {
            uint32_t id = (uint32_t)pos.size();
            pos.emplace_back(buf[3 + k * 3], buf[4 + k * 3], buf[5 + k * 3]);
            idx.push_back(id);
        }
    }
    std::fprintf(stderr, "loaded tris=%u verts=%zu\n", ntri, pos.size());

    // Build the half-edge mesh and verify one-ring traversal is available.
    orange::geometry::HalfEdgeMesh he;
    bool ok = he.build(pos, idx);
    std::fprintf(stderr, "halfedge build=%d welded_verts=%zu half_edges=%zu\n",
                 (int)ok, he.vertexCount(), he.halfEdgeCount());
    if (ok) {
        size_t total = 0, isolated = 0, boundary = 0, minR = 1e9, maxR = 0;
        std::vector<uint32_t> ring;
        for (uint32_t v = 0; v < (uint32_t)he.vertexCount(); ++v) {
            he.oneRing(v, ring);
            if (ring.empty()) { ++isolated; continue; }
            total += ring.size();
            if (ring.size() < minR) minR = ring.size();
            if (ring.size() > maxR) maxR = ring.size();
            if (he.isBoundary(v)) ++boundary;
        }
        size_t valid = he.vertexCount() - isolated;
        std::fprintf(stderr,
                     "one-ring: avg=%.2f min=%zu max=%zu isolated=%zu boundary=%zu\n",
                     valid ? (double)total / (double)valid : 0.0, minR, maxR, isolated, boundary);
    }

    orange::geometry::OcclusalPlaneParams params;
    if (argc >= 3) params.cuspSmoothIters = atoi(argv[2]);
    if (argc >= 4) params.cuspHeightGate = (float)atof(argv[3]);
    auto cusps = orange::geometry::findCusps(pos, idx, params);
    std::fprintf(stderr, "cusps=%zu\n", cusps.size());

    // 3D tooth segmentation (option C).
    {
        orange::geometry::ToothSegParams tp;
        if (argc >= 6) tp.prominence = (float)atof(argv[5]);
        if (argc >= 7) tp.heightGate = (float)atof(argv[6]);
        if (argc >= 8) tp.curvThresh = (float)atof(argv[7]);
        auto seg = orange::geometry::segmentTeeth(pos, idx, tp);
        size_t labeled = 0;
        for (int l : seg.label) if (l >= 0) ++labeled;
        std::fprintf(stderr, "3d-seg prom=%.2f gate=%.2f teeth=%d labeled=%.0f%%\n",
                     tp.prominence, tp.heightGate, seg.regions,
                     100.0 * (double)labeled / (double)seg.label.size());

        // Per-region stats -> report file for analysis.
        int K = seg.regions;
        std::vector<long> cnt(K, 0);
        std::vector<Eigen::Vector3f> mn(K, Eigen::Vector3f::Constant(1e30f));
        std::vector<Eigen::Vector3f> mx(K, Eigen::Vector3f::Constant(-1e30f));
        std::vector<Eigen::Vector3f> cen(K, Eigen::Vector3f::Zero());
        for (size_t i = 0; i < pos.size(); ++i) {
            int l = seg.label[i];
            if (l < 0 || l >= K) continue;
            ++cnt[l]; cen[l] += pos[i];
            mn[l] = mn[l].cwiseMin(pos[i]); mx[l] = mx[l].cwiseMax(pos[i]);
        }
        std::ofstream rep("seg_report.txt");
        rep << "label count diag(mm) sx sy sz cx cy cz\n";
        std::vector<float> diags;
        for (int l = 0; l < K; ++l) {
            if (cnt[l] == 0) continue;
            Eigen::Vector3f ext = mx[l] - mn[l];
            Eigen::Vector3f c = cen[l] / (float)cnt[l];
            float diag = ext.norm();
            diags.push_back(diag);
            rep << l << ' ' << cnt[l] << ' ' << diag << ' ' << ext.x() << ' ' << ext.y()
                << ' ' << ext.z() << ' ' << c.x() << ' ' << c.y() << ' ' << c.z() << '\n';
        }
        std::sort(diags.begin(), diags.end());
        rep << "# regions=" << diags.size();
        if (!diags.empty())
            rep << " diag_min=" << diags.front() << " diag_med=" << diags[diags.size()/2]
                << " diag_max=" << diags.back();
        rep << '\n';
        rep.close();
        std::fprintf(stderr, "wrote seg_report.txt (%zu regions)\n", diags.size());
    }

    // Segmentation region count via the occlusal channels (plane = cusp-fit PCA).
    if (cusps.size() >= 3) {
        auto plane = orange::geometry::wholeMeshPCA(cusps);
        float segProm = argc >= 5 ? (float)atof(argv[4]) : 0.45f;
        auto ch = orange::geometry::renderOcclusalChannels(pos, plane.position, plane.normal, 256, segProm);
        if (ch.valid) {
            std::set<uint32_t> labels;
            for (size_t p = 0; p < ch.segment.size(); p += 4) {
                if (ch.segment[p + 3] == 0) continue;  // transparent
                uint32_t key = (ch.segment[p] << 16) | (ch.segment[p + 1] << 8) | ch.segment[p + 2];
                labels.insert(key);
            }
            std::fprintf(stderr, "segment regions=%zu\n", labels.size());

            // SAM (on-device): normal image + cusp prompts.
            std::fprintf(stderr, "sam available=%d\n", (int)orange::ml::samAvailable());
            if (orange::ml::samAvailable()) {
                std::vector<uint8_t> rgbimg((size_t)ch.width * ch.height * 3);
                for (size_t i = 0; i < (size_t)ch.width * ch.height; ++i) {
                    rgbimg[i * 3 + 0] = ch.normal[i * 4 + 0];
                    rgbimg[i * 3 + 1] = ch.normal[i * 4 + 1];
                    rgbimg[i * 3 + 2] = ch.normal[i * 4 + 2];
                }
                std::vector<std::pair<float, float>> pts;
                for (auto& cu : cusps) {
                    Eigen::Vector3f d = cu - ch.origin;
                    float pxc = (d.dot(ch.u) - ch.aOff) / ch.cell;
                    float row = (float)(ch.height - 1) - (d.dot(ch.v) - ch.bOff) / ch.cell;
                    if (pxc >= 0 && pxc < ch.width && row >= 0 && row < ch.height)
                        pts.emplace_back(pxc, row);
                }
                auto sam = orange::ml::samSegment(rgbimg, ch.width, ch.height, pts);
                std::fprintf(stderr, "sam prompts=%zu valid=%d teeth=%d\n",
                             pts.size(), (int)sam.valid, sam.regions);
            }
        }
    }
    return 0;
}
