// Minimal consumer of the Orange binary SDK: no engine source in this project,
// just <orange/core/...> headers and a link against the prebuilt orange::core.
//
// Exercises the CPU-side toolkit headlessly (no window, no render plugin):
//   primitives  -> a sphere triangle soup
//   mesh_generation -> points back into a surface (TSDF + dual contouring)
//   kdtree      -> nearest-neighbour query on the cloud

#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Core>

#include <orange/core/kdtree.h>
#include <orange/core/mesh_generation.h>
#include <orange/core/primitives.h>

int main() {
    using namespace orange::geometry;

    // 1. Parametric primitive (the Create menu's builder, as a plain function).
    const std::vector<Triangle> sphere =
        buildSphere(10.0f, 48, Eigen::Vector3f(1.0f, 0.6f, 0.1f));
    std::printf("sphere: %zu triangles\n", sphere.size());

    // 2. Its vertices as an oriented point cloud.
    std::vector<Eigen::Vector3f> points, normals, colors;
    points.reserve(sphere.size() * 3);
    for (const Triangle& t : sphere) {
        for (int i = 0; i < 3; ++i) {
            points.push_back(t.v[i]);
            normals.push_back(t.n[i]);
            colors.push_back(t.c[i]);
        }
    }
    std::printf("cloud: %zu points\n", points.size());

    // 3. points in -> mesh out (voxel size in the cloud's own units).
    const std::vector<Triangle> rebuilt = pointsToMesh(points, normals, colors, 0.5f);
    std::printf("reconstructed: %zu triangles\n", rebuilt.size());

    // 4. Spatial query.
    KDTree tree;
    tree.build(points);
    float d2 = 0.0f;
    const int hit = tree.nearest(Eigen::Vector3f(10.0f, 0.0f, 0.0f), &d2);
    std::printf("nearest to (10,0,0): idx %d at distance %f\n", hit, std::sqrt(d2));

    return rebuilt.empty() ? 1 : 0;
}
