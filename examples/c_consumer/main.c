/* Pure C consumer of orange_c.dll -- compiled as C, not C++.
 *
 * This is the toolchain-independent path: one header (orange/c/orange.h, which
 * includes only <stdint.h>), one import library, one DLL. No Eigen, no STL, no
 * SDL headers, no matching-MSVC-version requirement.
 *
 * Build with CMake (see CMakeLists.txt), or directly:
 *   cl /TC main.c /I <sdk>/include <sdk>/lib/orange_c.lib
 */
#include <stdio.h>

#include <orange/c/orange.h>

int main(void) {
    const unsigned v = orangeVersion();
    printf("orange %u.%u.%u, C API v%u\n", (v >> 16) & 0xFFu, (v >> 8) & 0xFFu, v & 0xFFu,
           orangeCApiVersion());

    /* 1. A parametric primitive -> triangle soup. */
    const float orange_rgb[3] = {1.0f, 0.6f, 0.1f};
    OrangeMesh sphere = orangeBuildSphere(10.0f, 48, orange_rgb);
    if (!sphere) {
        printf("buildSphere failed: %s\n", orangeLastError());
        return 1;
    }
    const int triCount = orangeMeshTriangleCount(sphere);
    printf("sphere: %d triangles\n", triCount);

    /* 2. Its vertices as a point cloud (9 floats per triangle = 3 xyz verts). */
    OrangeVec3Array points = orangeVec3ArrayCreate(orangeMeshPositions(sphere), triCount * 3);
    printf("cloud: %d points\n", orangeVec3ArrayCount(points));

    /* 3. Normals + surface reconstruction back into a mesh. */
    OrangeVec3Array normals = orangeEstimateNormals(points, 16);
    OrangeMesh rebuilt = orangePointsToMesh(points, normals, NULL, 0.5f);
    if (!rebuilt) {
        printf("reconstruction failed: %s\n", orangeLastError());
        return 1;
    }
    printf("reconstructed: %d triangles\n", orangeMeshTriangleCount(rebuilt));

    {
        float lo[3], hi[3];
        if (orangeMeshBounds(rebuilt, lo, hi) == ORANGE_OK)
            printf("bounds: (%.2f %.2f %.2f) .. (%.2f %.2f %.2f)\n", lo[0], lo[1], lo[2], hi[0],
                   hi[1], hi[2]);
    }

    /* 4. Spatial query. */
    {
        OrangeKdTree tree = orangeKdTreeCreate(points);
        const float query[3] = {10.0f, 0.0f, 0.0f};
        float distance = 0.0f;
        const int nearest = orangeKdTreeNearest(tree, query, &distance);
        int neighbours[8];
        const int found = orangeKdTreeKNearest(tree, query, 8, neighbours, 8);
        printf("nearest to (10,0,0): idx %d at %f (%d neighbours returned)\n", nearest, distance,
               found);
        orangeKdTreeDestroy(tree);
    }

    /* 5. Every handle is freed explicitly; the library owns nothing else. */
    orangeVec3ArrayDestroy(normals);
    orangeVec3ArrayDestroy(points);
    orangeMeshDestroy(rebuilt);
    orangeMeshDestroy(sphere);
    return 0;
}
