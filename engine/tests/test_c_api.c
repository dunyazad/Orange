/* C ABI wrapper tests (orange/c/orange.h).
 *
 * Compiled as C and linked against the wrapper source directly (ORANGE_C_STATIC),
 * so the same entry points orange_c.dll exports are exercised without needing the
 * DLL to load -- useful on machines where unsigned DLLs are blocked, and it keeps
 * the wrapper honest in CI. Registered with CTest as orange_c_tests.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <orange/c/orange.h>

static int failures = 0;

static void check(int condition, const char* what) {
    if (!condition) {
        printf("  FAIL: %s (last error: %s)\n", what, orangeLastError());
        ++failures;
    }
}

static void testVersion(void) {
    printf("version...\n");
    check(orangeCApiVersion() == ORANGE_C_API_VERSION, "C API version matches the header");
    check(orangeVersion() != 0u, "engine version is set");
}

static void testVec3Array(void) {
    const float xyz[9] = {0, 0, 0, 1, 2, 3, -4, 5, 6};
    OrangeVec3Array arr = orangeVec3ArrayCreate(xyz, 3);
    float copy[9];
    printf("vec3 array...\n");
    check(arr != NULL, "create");
    check(orangeVec3ArrayCount(arr) == 3, "count");
    check(orangeVec3ArrayData(arr) != NULL, "data pointer");
    check(orangeVec3ArrayData(arr)[4] == 2.0f, "data round-trips");
    check(orangeVec3ArrayCopy(arr, copy, 9) == 9, "copy returns float count");
    check(memcmp(copy, xyz, sizeof(xyz)) == 0, "copy matches the input");
    check(orangeVec3ArrayCopy(arr, copy, 8) == ORANGE_ERROR_TOO_SMALL, "short buffer rejected");
    orangeVec3ArrayDestroy(arr);

    /* NULL handling must never crash. */
    check(orangeVec3ArrayCount(NULL) == ORANGE_ERROR_INVALID, "NULL count is an error");
    check(orangeVec3ArrayData(NULL) == NULL, "NULL data is NULL");
    orangeVec3ArrayDestroy(NULL);
}

static void testPrimitivesAndMesh(void) {
    const float color[3] = {1.0f, 0.6f, 0.1f};
    OrangeMesh sphere = orangeBuildSphere(10.0f, 32, color);
    float lo[3], hi[3];
    int tris;
    printf("primitives + mesh...\n");
    check(sphere != NULL, "buildSphere");
    tris = orangeMeshTriangleCount(sphere);
    check(tris > 0, "triangle count");
    check(orangeMeshPositions(sphere) != NULL, "positions");
    check(orangeMeshNormals(sphere) != NULL, "normals");
    check(orangeMeshColors(sphere) != NULL, "colors");
    check(orangeMeshColors(sphere)[0] == color[0], "vertex color is what was asked for");
    check(orangeMeshBounds(sphere, lo, hi) == ORANGE_OK, "bounds");
    check(fabsf(hi[0] - 10.0f) < 0.5f && fabsf(lo[0] + 10.0f) < 0.5f, "bounds match the radius");
    orangeMeshDestroy(sphere);

    check(orangeMeshTriangleCount(NULL) == ORANGE_ERROR_INVALID, "NULL mesh is an error");
    orangeMeshDestroy(NULL);
}

static void testReconstruction(void) {
    const float color[3] = {1, 1, 1};
    OrangeMesh sphere = orangeBuildSphere(10.0f, 32, color);
    const int tris = orangeMeshTriangleCount(sphere);
    OrangeVec3Array points = orangeVec3ArrayCreate(orangeMeshPositions(sphere), tris * 3);
    OrangeVec3Array normals = orangeEstimateNormals(points, 16);
    OrangeMesh rebuilt = orangePointsToMesh(points, normals, NULL, 0.5f);
    OrangeVec3Array smoothed = orangeSmoothPoints(points, 2, 0.5f, 1, 12);
    printf("normals + reconstruction + smoothing...\n");
    check(normals != NULL, "estimateNormals");
    check(orangeVec3ArrayCount(normals) == orangeVec3ArrayCount(points), "one normal per point");
    check(rebuilt != NULL, "pointsToMesh");
    check(orangeMeshTriangleCount(rebuilt) > 0, "reconstruction produced triangles");
    check(smoothed != NULL, "smoothPoints");
    check(orangeVec3ArrayCount(smoothed) == orangeVec3ArrayCount(points), "smoothing keeps count");

    /* Empty / NULL inputs must fail cleanly with a message, not crash. */
    check(orangePointsToMesh(NULL, NULL, NULL, 0.5f) == NULL, "NULL points rejected");
    check(orangeLastError()[0] != '\0', "error message recorded");

    orangeVec3ArrayDestroy(smoothed);
    orangeVec3ArrayDestroy(normals);
    orangeVec3ArrayDestroy(points);
    orangeMeshDestroy(rebuilt);
    orangeMeshDestroy(sphere);
}

static void testKdTree(void) {
    const float xyz[12] = {0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5};
    const float query[3] = {4.6f, 0.0f, 0.0f};
    OrangeVec3Array points = orangeVec3ArrayCreate(xyz, 4);
    OrangeKdTree tree = orangeKdTreeCreate(points);
    float distance = -1.0f;
    int indices[4];
    int n;
    printf("kd-tree...\n");
    check(tree != NULL, "create");
    check(orangeKdTreeNearest(tree, query, &distance) == 1, "nearest index");
    check(fabsf(distance - 0.4f) < 1e-4f, "nearest distance is euclidean, not squared");
    n = orangeKdTreeKNearest(tree, query, 3, indices, 4);
    check(n == 3, "kNearest count");
    check(indices[0] == 1, "kNearest is sorted nearest-first");
    check(orangeKdTreeRadius(tree, query, 100.0f, NULL, 0) == 4, "radius count-only probe");
    check(orangeKdTreeRadius(tree, query, 1.0f, indices, 4) == 1, "radius query");
    check(orangeKdTreeNearest(NULL, query, NULL) == ORANGE_ERROR_INVALID, "NULL tree is an error");

    orangeKdTreeDestroy(tree);
    orangeVec3ArrayDestroy(points);
}

static void testIcp(void) {
    /* Translate a cloud, then let ICP recover the offset. */
    float src[30], dst[30];
    OrangeVec3Array a, b;
    float m[16], rmse = -1.0f;
    int iters = -1, i;
    printf("icp...\n");
    for (i = 0; i < 10; ++i) {
        src[i * 3 + 0] = (float)i;
        src[i * 3 + 1] = (float)(i * i) * 0.1f;
        src[i * 3 + 2] = 0.0f;
        dst[i * 3 + 0] = src[i * 3 + 0] + 0.3f;
        dst[i * 3 + 1] = src[i * 3 + 1];
        dst[i * 3 + 2] = src[i * 3 + 2];
    }
    a = orangeVec3ArrayCreate(src, 10);
    b = orangeVec3ArrayCreate(dst, 10);
    check(orangeIcpAlign(a, b, 30, m, &rmse, &iters) == ORANGE_OK, "icpAlign");
    check(rmse >= 0.0f && iters > 0, "rmse / iteration count filled in");
    /* Column-major 4x4: translation lives in elements 12..14. */
    check(fabsf(m[12] - 0.3f) < 0.15f, "recovered translation (column-major layout)");
    check(orangeIcpAlign(NULL, b, 10, m, NULL, NULL) == ORANGE_ERROR_INVALID, "NULL src rejected");
    orangeVec3ArrayDestroy(a);
    orangeVec3ArrayDestroy(b);
}

static void testIoErrors(void) {
    printf("io errors...\n");
    check(orangePointsLoadFile("does_not_exist.ply") == NULL, "missing file fails");
    check(orangeLastError()[0] != '\0', "error message recorded");
    check(orangePointsLoadFile("model.unsupported") == NULL, "unknown extension fails");
    check(orangePointsLoadFile(NULL) == NULL, "NULL path fails");
    check(orangeReconstructFromFile(NULL, 0.5f) == NULL, "NULL path fails (reconstruct)");
}

int main(void) {
    testVersion();
    testVec3Array();
    testPrimitivesAndMesh();
    testReconstruction();
    testKdTree();
    testIcp();
    testIoErrors();

    if (failures == 0) {
        printf("all C ABI tests passed\n");
        return 0;
    }
    printf("%d C ABI test(s) failed\n", failures);
    return 1;
}
