#include "orange/core/poisson_reconstruction.h"

#include <algorithm>
#include <cmath>
#include <execution>
#include <numeric>
#include <utility>

#include "orange/core/marching_cubes_tables.h"

namespace orange::geometry {

namespace {

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline int   clampi(int v, int lo, int hi)       { return v < lo ? lo : (v > hi ? hi : v); }

// Classic marching-cubes corner offsets (matches Paul Bourke's kEdgeTable/kTriTable).
constexpr int kCorner[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
    {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1},
};
// Edge -> its two corner endpoints.
constexpr int kEdge[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
    {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

// --- Multigrid helpers (vertex-centered, grid dim = 2^depth + 1) -------------
inline size_t gI(int x, int y, int z, int N) { return (size_t)(z * N + y) * N + x; }

// Red-black Gauss-Seidel for L(u) = f, where L(u) = (6u - sum_nbrs)/h^2 + lambda*w*u.
// Solving a node: u = (f*h^2 + sum_nbrs) / (6 + lambda*w*h^2). Boundary nodes are
// fixed at 0 (skipped). Parallel across interior z-slices within each color.
void mgSmooth(std::vector<float>& u, const std::vector<float>& f, const std::vector<float>& w,
              int N, float h, float lambda, int iters) {
    if (N < 3) return;
    const float h2 = h * h;
    std::vector<int> zs(N - 2);
    std::iota(zs.begin(), zs.end(), 1);
    for (int it = 0; it < iters; ++it)
        for (int color = 0; color < 2; ++color)
            std::for_each(std::execution::par, zs.begin(), zs.end(), [&](int z) {
                for (int y = 1; y < N - 1; ++y)
                    for (int x = 1; x < N - 1; ++x) {
                        if (((x + y + z) & 1) != color) continue;
                        size_t k = gI(x, y, z, N);
                        float sum = u[gI(x + 1, y, z, N)] + u[gI(x - 1, y, z, N)] +
                                    u[gI(x, y + 1, z, N)] + u[gI(x, y - 1, z, N)] +
                                    u[gI(x, y, z + 1, N)] + u[gI(x, y, z - 1, N)];
                        u[k] = (f[k] * h2 + sum) / (6.0f + lambda * w[k] * h2);
                    }
            });
}

// Full-weighting restriction fine (Nf=2*(Nc-1)+1) -> coarse (Nc). Boundary coarse
// nodes stay 0 (matches the smoother's fixed boundary).
void mgRestrict(const std::vector<float>& fine, int Nf, std::vector<float>& coarse, int Nc) {
    if (Nc < 3) return;
    std::vector<int> zs(Nc - 2);
    std::iota(zs.begin(), zs.end(), 1);
    std::for_each(std::execution::par, zs.begin(), zs.end(), [&](int Z) {
        for (int Y = 1; Y < Nc - 1; ++Y)
            for (int X = 1; X < Nc - 1; ++X) {
                int xf = 2 * X, yf = 2 * Y, zf = 2 * Z;
                float s = 0.0f;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            float wgt = (dx ? 0.5f : 1.0f) * (dy ? 0.5f : 1.0f) *
                                        (dz ? 0.5f : 1.0f) / 8.0f;
                            s += wgt * fine[gI(xf + dx, yf + dy, zf + dz, Nf)];
                        }
                coarse[gI(X, Y, Z, Nc)] = s;
            }
    });
}

// Trilinear prolongation coarse (Nc) -> fine (Nf), overwriting `fine`. Even fine
// indices coincide with a coarse node; odd ones average the two neighbors.
void mgProlong(const std::vector<float>& coarse, int Nc, std::vector<float>& fine, int Nf) {
    std::vector<int> zs(Nf);
    std::iota(zs.begin(), zs.end(), 0);
    std::for_each(std::execution::par, zs.begin(), zs.end(), [&](int z) {
        int Z = z >> 1, Z1 = (z & 1) ? std::min(Z + 1, Nc - 1) : Z; float fz = (z & 1) ? 0.5f : 0.0f;
        for (int y = 0; y < Nf; ++y) {
            int Y = y >> 1, Y1 = (y & 1) ? std::min(Y + 1, Nc - 1) : Y; float fy = (y & 1) ? 0.5f : 0.0f;
            for (int x = 0; x < Nf; ++x) {
                int X = x >> 1, X1 = (x & 1) ? std::min(X + 1, Nc - 1) : X; float fx = (x & 1) ? 0.5f : 0.0f;
                float v =
                    (1 - fx) * (1 - fy) * (1 - fz) * coarse[gI(X, Y, Z, Nc)] +
                    fx * (1 - fy) * (1 - fz) * coarse[gI(X1, Y, Z, Nc)] +
                    (1 - fx) * fy * (1 - fz) * coarse[gI(X, Y1, Z, Nc)] +
                    fx * fy * (1 - fz) * coarse[gI(X1, Y1, Z, Nc)] +
                    (1 - fx) * (1 - fy) * fz * coarse[gI(X, Y, Z1, Nc)] +
                    fx * (1 - fy) * fz * coarse[gI(X1, Y, Z1, Nc)] +
                    (1 - fx) * fy * fz * coarse[gI(X, Y1, Z1, Nc)] +
                    fx * fy * fz * coarse[gI(X1, Y1, Z1, Nc)];
                fine[gI(x, y, z, Nf)] = v;
            }
        }
    });
}

} // namespace

std::vector<Triangle> poissonReconstruct(const std::vector<Eigen::Vector3f>& points,
                                         const std::vector<Eigen::Vector3f>& normals,
                                         const PoissonParams& params,
                                         const std::function<void(float)>& progress) {
    std::vector<Triangle> tris;
    const size_t N = points.size();
    if (N < 4 || normals.size() != N) return tris;

    auto report = [&](float f) { if (progress) progress(clampf(f, 0.0f, 1.0f)); };

    // 1) Cubic bounds, padded by `scale` so the surface has room to close.
    Eigen::Vector3f mn = points[0], mx = points[0];
    for (const auto& p : points) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    Eigen::Vector3f center = 0.5f * (mn + mx);
    float ext = (mx - mn).maxCoeff();
    if (ext <= 0.0f) return tris;
    float side = ext * std::max(1.0f, params.scale);
    Eigen::Vector3f org = center - Eigen::Vector3f::Constant(side * 0.5f);

    const int   depth = clampi(params.depth, 3, 14);
    const int   dim   = (1 << depth) + 1;    // grid samples per axis (2^depth+1, for multigrid)
    const float h     = side / float(dim - 1);
    const size_t M    = (size_t)dim * dim * dim;
    auto idx = [dim](int x, int y, int z) { return (size_t)(z * dim + y) * dim + x; };
    auto toGrid = [&](const Eigen::Vector3f& p) { return (p - org) / h; };

    // 2) Splat oriented normals into a vector field V, and a point-density W
    //    (the screening weight). Trilinear distribution to the 8 nearest nodes.
    std::vector<float> Vx(M, 0.0f), Vy(M, 0.0f), Vz(M, 0.0f), W(M, 0.0f);
    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3f g = toGrid(points[i]);
        int x0 = (int)std::floor(g.x()), y0 = (int)std::floor(g.y()), z0 = (int)std::floor(g.z());
        float fx = g.x() - x0, fy = g.y() - y0, fz = g.z() - z0;
        const Eigen::Vector3f& n = normals[i];
        for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    int X = x0 + dx, Y = y0 + dy, Z = z0 + dz;
                    if (X < 0 || Y < 0 || Z < 0 || X >= dim || Y >= dim || Z >= dim) continue;
                    float w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
                    size_t k = idx(X, Y, Z);
                    Vx[k] += w * n.x(); Vy[k] += w * n.y(); Vz[k] += w * n.z(); W[k] += w;
                }
        if ((i & 0x3FFF) == 0) report(0.02f + 0.16f * float(i) / float(N));
    }
    report(0.18f);

    // Interior z-slices, used to drive the parallel sweeps below.
    std::vector<int> zs(dim - 2 > 0 ? dim - 2 : 0);
    std::iota(zs.begin(), zs.end(), 1);

    // 3) Right-hand side b = div(V), central differences (interior nodes only).
    //    Each z-slice is independent -> parallel.
    std::vector<float> b(M, 0.0f);
    const float invH2 = 0.5f / h;  // central diff: (f[+1]-f[-1]) / (2h)
    std::for_each(std::execution::par, zs.begin(), zs.end(), [&](int z) {
        for (int y = 1; y < dim - 1; ++y)
            for (int x = 1; x < dim - 1; ++x) {
                float d = (Vx[idx(x + 1, y, z)] - Vx[idx(x - 1, y, z)]) +
                          (Vy[idx(x, y + 1, z)] - Vy[idx(x, y - 1, z)]) +
                          (Vz[idx(x, y, z + 1)] - Vz[idx(x, y, z - 1)]);
                b[idx(x, y, z)] = d * invH2;
            }
    });
    report(0.25f);

    // V is consumed by the divergence; free it before the solve so deep grids fit
    // (peak is 5 grids during divergence, not 6 for the whole run).
    std::vector<float>().swap(Vx);
    std::vector<float>().swap(Vy);
    std::vector<float>().swap(Vz);

    // 4) Solve (Laplacian - lambda*W) chi = b by red-black Gauss-Seidel. Within a
    //    color, every node only reads its 6 opposite-color neighbors, so a color
    //    sweep is data-parallel across slices. Discretized:
    //    (sum_neighbors(chi) - 6 chi)/h^2 - lambda W chi = b
    //  => chi = (sum_neighbors - b h^2) / (6 + lambda W h^2).
    const float lambda = std::max(0.0f, params.pointWeight);
    const int   iters  = clampi(params.iterations, 1, 200);

    std::vector<int>   dims;          // dims[0] = dim (fine), halving down to <= 4
    std::vector<float> hs;
    for (int l = 0;; ++l) {
        int Nl = ((dim - 1) >> l) + 1;
        dims.push_back(Nl);
        hs.push_back(h * float(1 << l));
        if (Nl <= 4) break;
    }
    const int Lv = (int)dims.size();
    auto vol = [](int Nn) { return (size_t)Nn * Nn * Nn; };

    // Per-level density (w) and right-hand side (f = -b), restricted from the fine
    // grid. Level 0 takes ownership of W and a fresh -b.
    std::vector<std::vector<float>> wlev(Lv), flev(Lv), ulev(Lv);
    wlev[0] = std::move(W);
    flev[0].resize(M);
    for (size_t k = 0; k < M; ++k) flev[0][k] = -b[k];
    std::vector<float>().swap(b);
    for (int l = 1; l < Lv; ++l) {
        wlev[l].assign(vol(dims[l]), 0.0f);
        flev[l].assign(vol(dims[l]), 0.0f);
        mgRestrict(wlev[l - 1], dims[l - 1], wlev[l], dims[l]);
        mgRestrict(flev[l - 1], dims[l - 1], flev[l], dims[l]);
    }

    // Cascadic sweep, coarse -> fine.
    ulev[Lv - 1].assign(vol(dims[Lv - 1]), 0.0f);
    mgSmooth(ulev[Lv - 1], flev[Lv - 1], wlev[Lv - 1], dims[Lv - 1], hs[Lv - 1], lambda,
             8 * dims[Lv - 1]);  // fully converge the tiny coarsest grid
    for (int l = Lv - 2; l >= 0; --l) {
        ulev[l].assign(vol(dims[l]), 0.0f);
        mgProlong(ulev[l + 1], dims[l + 1], ulev[l], dims[l]);   // initial guess from coarser level
        mgSmooth(ulev[l], flev[l], wlev[l], dims[l], hs[l], lambda, std::max(16, iters));
        std::vector<float>().swap(ulev[l + 1]);  // release coarser scratch once consumed
        std::vector<float>().swap(flev[l + 1]);
        std::vector<float>().swap(wlev[l + 1]);
        std::vector<float>().swap(flev[l]);
        report(0.25f + 0.55f * float(Lv - 1 - l) / float(Lv));
    }
    std::vector<float> chi = std::move(ulev[0]);
    std::vector<float>().swap(W);  // density no longer needed
    report(0.80f);

    // 5) Iso-value = mean of chi (trilinearly sampled) at the input points: the
    //    level set that best passes through the data.
    double isoSum = 0.0; size_t isoN = 0;
    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3f g = toGrid(points[i]);
        int x0 = (int)std::floor(g.x()), y0 = (int)std::floor(g.y()), z0 = (int)std::floor(g.z());
        if (x0 < 0 || y0 < 0 || z0 < 0 || x0 >= dim - 1 || y0 >= dim - 1 || z0 >= dim - 1) continue;
        float fx = g.x() - x0, fy = g.y() - y0, fz = g.z() - z0;
        float v = 0.0f;
        for (int dz = 0; dz < 2; ++dz)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    float w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
                    v += w * chi[idx(x0 + dx, y0 + dy, z0 + dz)];
                }
        isoSum += v; ++isoN;
    }
    const float iso = isoN ? (float)(isoSum / isoN) : 0.0f;

    // chi was built from outward normals, so grad(chi) ~ outward normal at the
    // surface. Central-difference node gradient (clamped at the border).
    auto nodeGrad = [&](int x, int y, int z) {
        auto cl = [&](int a) { return a < 0 ? 0 : (a >= dim ? dim - 1 : a); };
        return Eigen::Vector3f(chi[idx(cl(x + 1), y, z)] - chi[idx(cl(x - 1), y, z)],
                               chi[idx(x, cl(y + 1), z)] - chi[idx(x, cl(y - 1), z)],
                               chi[idx(x, y, cl(z + 1))] - chi[idx(x, y, cl(z - 1))]);
    };

    // 6) Marching cubes over the chi grid at `iso`. Each cell-slice z is
    //    independent; build per-slice triangle buffers in parallel, then merge.
    const Eigen::Vector3f kColor(0.78f, 0.80f, 0.83f);  // neutral; recolored by color mode
    std::vector<int> zc(dim - 1 > 0 ? dim - 1 : 0);
    std::iota(zc.begin(), zc.end(), 0);
    std::vector<std::vector<Triangle>> perZ(zc.size());
    std::for_each(std::execution::par, zc.begin(), zc.end(), [&](int z) {
        std::vector<Triangle>& local = perZ[z];
        Eigen::Vector3f cp[8], cg[8]; float cv[8];
        Eigen::Vector3f ev[12], en[12];
        for (int y = 0; y < dim - 1; ++y)
            for (int x = 0; x < dim - 1; ++x) {
                int ci = 0;
                for (int c = 0; c < 8; ++c) {
                    int X = x + kCorner[c][0], Y = y + kCorner[c][1], Z = z + kCorner[c][2];
                    cv[c] = chi[idx(X, Y, Z)];
                    cp[c] = org + Eigen::Vector3f(X * h, Y * h, Z * h);
                    cg[c] = nodeGrad(X, Y, Z);
                    if (cv[c] < iso) ci |= (1 << c);
                }
                int em = kEdgeTable[ci];
                if (em == 0) continue;
                for (int e = 0; e < 12; ++e) {
                    if (!(em & (1 << e))) continue;
                    int a = kEdge[e][0], bb = kEdge[e][1];
                    float va = cv[a], vb = cv[bb];
                    float t = std::fabs(vb - va) > 1e-7f ? (iso - va) / (vb - va) : 0.5f;
                    ev[e] = cp[a] + t * (cp[bb] - cp[a]);
                    Eigen::Vector3f gn = cg[a] + t * (cg[bb] - cg[a]);
                    en[e] = gn.squaredNorm() > 1e-12f ? gn.normalized() : Eigen::Vector3f::UnitY();
                }
                for (int t = 0; kTriTable[ci][t] != -1; t += 3) {
                    int e0 = kTriTable[ci][t], e1 = kTriTable[ci][t + 1], e2 = kTriTable[ci][t + 2];
                    Triangle tri;
                    tri.v[0] = ev[e0]; tri.v[1] = ev[e1]; tri.v[2] = ev[e2];
                    tri.n[0] = en[e0]; tri.n[1] = en[e1]; tri.n[2] = en[e2];
                    // Wind so the geometric normal agrees with the (outward) gradient.
                    Eigen::Vector3f fn = (tri.v[1] - tri.v[0]).cross(tri.v[2] - tri.v[0]);
                    if (fn.dot(tri.n[0] + tri.n[1] + tri.n[2]) < 0.0f) {
                        std::swap(tri.v[1], tri.v[2]);
                        std::swap(tri.n[1], tri.n[2]);
                    }
                    tri.c[0] = tri.c[1] = tri.c[2] = kColor;
                    local.push_back(tri);
                }
            }
    });
    report(0.95f);
    size_t total = 0;
    for (const auto& v : perZ) total += v.size();
    tris.reserve(total);
    for (auto& v : perZ) tris.insert(tris.end(), v.begin(), v.end());
    report(1.0f);
    return tris;
}

} // namespace orange::geometry
