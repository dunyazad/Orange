// Minimal mesh-file loaders for appOrange: Wavefront OBJ and STL (binary +
// ASCII). They fill a render::Vertex/index list. Since the mesh shader is
// unlit (FragColor = texture * color), each vertex's color is set from its
// surface normal (n*0.5+0.5) so the loaded shape reads clearly without lights.
#pragma once

#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <sstream>
#include <string>
#include <vector>

#include "orange/render/types.h"

namespace orange::meshio {

// Optional progress sink: called with a fraction in [0,1] as a load proceeds.
// May be empty. Loaders throttle calls to once per whole-percent change so a
// background thread updating an atomic stays cheap even for millions of points.
using ProgressFn = std::function<void(float)>;

struct V3 { float x, y, z; };

inline V3 sub(const V3& a, const V3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 cross(const V3& a, const V3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// Normalizes `n`, falling back to +Y for a degenerate (zero-length) vector.
inline V3 normOrUp(V3 n) {
    float L = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (L < 1e-12f) return {0.0f, 1.0f, 0.0f};
    return {n.x / L, n.y / L, n.z / L};
}
inline render::Vertex vert(const V3& p, const V3& n) {
    // Normal -> color in [0,1]; gives a shaded look under the unlit shader.
    return {{p.x, p.y, p.z}, {n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f}};
}

// Builds an indexed mesh from positions + triangle indices, giving each vertex
// the area-weighted average of its incident face normals (mapped to color).
// Triangles with out-of-range indices are skipped. Shared by the OBJ/PLY loaders.
// Reports `p` (0..1) through `cb` only when the whole-percent value changes, so
// per-element calls from a tight parse loop stay cheap. `cb` may be empty.
struct Throttle {
    const ProgressFn& cb;
    int last = -1;
    explicit Throttle(const ProgressFn& c) : cb(c) {}
    void operator()(float p) {
        if (!cb) return;
        int pc = static_cast<int>(p * 100.0f);
        if (pc != last) { last = pc; cb(p); }
    }
};

inline bool buildIndexed(const std::vector<V3>& pos, std::vector<uint32_t>& tris,
                         std::vector<render::Vertex>& outV, std::vector<uint32_t>& outI) {
    if (pos.empty() || tris.empty()) return false;
    std::vector<V3> nrm(pos.size(), V3{0, 0, 0});
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        uint32_t a = tris[t], b = tris[t + 1], c = tris[t + 2];
        if (a >= pos.size() || b >= pos.size() || c >= pos.size()) continue;
        V3 fn = cross(sub(pos[b], pos[a]), sub(pos[c], pos[a]));  // area-weighted
        for (uint32_t i : {a, b, c}) { nrm[i].x += fn.x; nrm[i].y += fn.y; nrm[i].z += fn.z; }
    }
    outV.resize(pos.size());
    for (size_t i = 0; i < pos.size(); ++i) outV[i] = vert(pos[i], normOrUp(nrm[i]));
    outI = std::move(tris);
    return true;
}

// --- Wavefront OBJ ---------------------------------------------------------
// Reads `v` positions and `f` faces (polygons fan-triangulated; vertex/uv/normal
// triplets accepted, only the position index is used). Per-vertex normals are
// area-weighted averages of the incident faces.
inline bool loadObj(const char* path, std::vector<render::Vertex>& outV,
                    std::vector<uint32_t>& outI, const ProgressFn& progress = {}) {
    std::ifstream f(path);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff total = f.tellg();      // file size (bytes) for progress
    f.seekg(0, std::ios::beg);
    Throttle report(progress);

    std::vector<V3>       pos;
    std::vector<uint32_t> tris;
    std::string           line;
    std::streamoff        read = 0;
    while (std::getline(f, line)) {
        read += static_cast<std::streamoff>(line.size()) + 1;  // +1 ~ newline
        if (total > 0) report(0.95f * static_cast<float>(read) / static_cast<float>(total));
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            const char* s = line.c_str() + 2;
            char*       e = nullptr;
            V3          p{};
            p.x = std::strtof(s, &e); if (e == s) continue; s = e;
            p.y = std::strtof(s, &e); if (e == s) continue; s = e;
            p.z = std::strtof(s, &e); if (e == s) continue;
            pos.push_back(p);
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::vector<int> idx;
            const char*      p = line.c_str() + 1;
            while (*p) {
                while (*p == ' ' || *p == '\t') ++p;
                if (!*p) break;
                int vi = std::atoi(p);  // first int of the "v/vt/vn" token
                if (vi < 0) vi = static_cast<int>(pos.size()) + vi + 1;  // relative
                if (vi >= 1 && vi <= static_cast<int>(pos.size())) idx.push_back(vi - 1);
                while (*p && *p != ' ' && *p != '\t') ++p;
            }
            for (size_t k = 1; k + 1 < idx.size(); ++k) {  // triangle fan
                tris.push_back(static_cast<uint32_t>(idx[0]));
                tris.push_back(static_cast<uint32_t>(idx[k]));
                tris.push_back(static_cast<uint32_t>(idx[k + 1]));
            }
        }
    }
    bool ok = buildIndexed(pos, tris, outV, outI);
    if (progress) progress(1.0f);
    return ok;
}

// --- STL -------------------------------------------------------------------
// Binary STL: 80-byte header, uint32 triangle count, then 50 bytes per triangle
// (normal + 3 vertices + attribute). Vertices are not shared.
inline bool loadStlBinary(const std::vector<uint8_t>& data,
                          std::vector<render::Vertex>& outV,
                          std::vector<uint32_t>& outI, const ProgressFn& progress = {}) {
    if (data.size() < 84) return false;
    uint32_t n = 0;
    std::memcpy(&n, &data[80], 4);
    if (data.size() < 84 + static_cast<size_t>(n) * 50) return false;

    Throttle report(progress);
    outV.reserve(static_cast<size_t>(n) * 3);
    outI.reserve(static_cast<size_t>(n) * 3);
    const uint8_t* p = &data[84];
    for (uint32_t i = 0; i < n; ++i) {
        if (n > 0) report(static_cast<float>(i) / static_cast<float>(n));
        float fn[3], v[9];
        std::memcpy(fn, p, 12);
        std::memcpy(v, p + 12, 36);
        V3 nrm{fn[0], fn[1], fn[2]};
        if (nrm.x == 0 && nrm.y == 0 && nrm.z == 0)  // some exporters leave it zero
            nrm = cross(sub({v[3], v[4], v[5]}, {v[0], v[1], v[2]}),
                        sub({v[6], v[7], v[8]}, {v[0], v[1], v[2]}));
        nrm = normOrUp(nrm);
        for (int k = 0; k < 3; ++k) {
            outI.push_back(static_cast<uint32_t>(outV.size()));
            outV.push_back(vert({v[k * 3], v[k * 3 + 1], v[k * 3 + 2]}, nrm));
        }
        p += 50;
    }
    if (progress) progress(1.0f);
    return !outV.empty();
}

// ASCII STL: scans for "vertex x y z" triplets; every 3 makes a triangle.
inline bool loadStlAscii(const std::string& text, std::vector<render::Vertex>& outV,
                         std::vector<uint32_t>& outI, const ProgressFn& progress = {}) {
    Throttle        report(progress);
    std::vector<V3> vs;
    size_t          pos = 0;
    while ((pos = text.find("vertex", pos)) != std::string::npos) {
        if (!text.empty()) report(0.95f * static_cast<float>(pos) / static_cast<float>(text.size()));
        V3 p{};
        if (std::sscanf(text.c_str() + pos + 6, "%f %f %f", &p.x, &p.y, &p.z) == 3)
            vs.push_back(p);
        pos += 6;
    }
    if (vs.size() < 3) return false;
    for (size_t i = 0; i + 2 < vs.size(); i += 3) {
        V3 nrm = normOrUp(cross(sub(vs[i + 1], vs[i]), sub(vs[i + 2], vs[i])));
        for (int k = 0; k < 3; ++k) {
            outI.push_back(static_cast<uint32_t>(outV.size()));
            outV.push_back(vert(vs[i + k], nrm));
        }
    }
    if (progress) progress(1.0f);
    return !outV.empty();
}

inline bool loadStl(const char* path, std::vector<render::Vertex>& outV,
                    std::vector<uint32_t>& outI, const ProgressFn& progress = {}) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    if (data.empty()) return false;
    // Trust the binary triangle count when the file size matches exactly;
    // otherwise treat it as ASCII.
    if (data.size() >= 84) {
        uint32_t n = 0;
        std::memcpy(&n, &data[80], 4);
        if (data.size() == 84 + static_cast<size_t>(n) * 50)
            return loadStlBinary(data, outV, outI, progress);
    }
    return loadStlAscii(std::string(data.begin(), data.end()), outV, outI, progress);
}

// --- PLY -------------------------------------------------------------------
// Reads vertex x/y/z and face index lists (fan-triangulated); other vertex
// properties are skipped. Supports ASCII and binary_little_endian. Normals are
// computed (area-weighted), matching the other loaders.
inline bool loadPly(const char* path, std::vector<render::Vertex>& outV,
                    std::vector<uint32_t>& outI, const ProgressFn& progress = {}) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    Throttle report(progress);

    // Slurp the whole file and parse from memory: a multi-million-point PLY does
    // one read instead of tens of millions of per-value stream reads. A trailing
    // NUL lets strtod/strtoll scan the ASCII body without overrunning the buffer.
    std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    if (buf.empty()) return false;
    buf.push_back('\0');
    const char* cur = buf.data();
    const char* end = cur + buf.size() - 1;  // exclude the sentinel NUL

    auto headerLine = [&](std::string& s) -> bool {
        if (cur >= end) return false;
        const char* nl = static_cast<const char*>(std::memchr(cur, '\n', end - cur));
        const char* le = nl ? nl : end;
        s.assign(cur, le);
        while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
        cur = nl ? nl + 1 : end;
        return true;
    };
    std::string line;
    if (!headerLine(line) || line.rfind("ply", 0) != 0) return false;

    enum Fmt { ASCII, BIN_LE, BIN_BE } fmt = ASCII;
    struct Prop { int kind = 2, size = 4, countKind = 1, countSize = 1; bool isList = false; };
    // kind: 0 = signed int, 1 = unsigned int, 2 = float.
    auto mapType = [](const std::string& t, int& kind, int& size) {
        if (t == "char" || t == "int8")          { kind = 0; size = 1; }
        else if (t == "uchar" || t == "uint8")   { kind = 1; size = 1; }
        else if (t == "short" || t == "int16")   { kind = 0; size = 2; }
        else if (t == "ushort" || t == "uint16") { kind = 1; size = 2; }
        else if (t == "int" || t == "int32")     { kind = 0; size = 4; }
        else if (t == "uint" || t == "uint32")   { kind = 1; size = 4; }
        else if (t == "double" || t == "float64"){ kind = 2; size = 8; }
        else                                     { kind = 2; size = 4; }  // float
    };

    int               vertexCount = 0, faceCount = 0, ix = -1, iy = -1, iz = -1;
    int               inx = -1, iny = -1, inz = -1;  // normals (optional)
    int               ir = -1, ig = -1, ib = -1;     // colors (optional)
    std::vector<Prop> vprops;
    Prop              faceProp;
    bool              haveFace = false;
    std::string       curElem;
    while (headerLine(line)) {
        std::istringstream ss(line);
        std::string        kw;
        ss >> kw;
        if (kw == "format") {
            std::string ff; ss >> ff;
            fmt = ff == "binary_little_endian" ? BIN_LE
                : ff == "binary_big_endian"    ? BIN_BE
                                               : ASCII;
        } else if (kw == "element") {
            std::string en; int n = 0; ss >> en >> n; curElem = en;
            if (en == "vertex") vertexCount = n; else if (en == "face") faceCount = n;
        } else if (kw == "property") {
            std::string t1; ss >> t1;
            if (t1 == "list") {
                std::string ct, it, nm; ss >> ct >> it >> nm;
                if (curElem == "face") {
                    faceProp.isList = true;
                    mapType(ct, faceProp.countKind, faceProp.countSize);
                    mapType(it, faceProp.kind, faceProp.size);
                    haveFace = true;
                }
            } else if (curElem == "vertex") {
                std::string nm; ss >> nm;
                Prop p; mapType(t1, p.kind, p.size); vprops.push_back(p);
                int idx = static_cast<int>(vprops.size()) - 1;
                if (nm == "x") ix = idx;
                else if (nm == "y") iy = idx;
                else if (nm == "z") iz = idx;
                else if (nm == "nx") inx = idx;
                else if (nm == "ny") iny = idx;
                else if (nm == "nz") inz = idx;
                else if (nm == "red")   ir = idx;
                else if (nm == "green") ig = idx;
                else if (nm == "blue")  ib = idx;
            }
        } else if (kw == "end_header") {
            break;
        }
    }
    if (vertexCount <= 0 || ix < 0 || iy < 0 || iz < 0 || fmt == BIN_BE) return false;

    // Binary readers advance `cur` through the in-memory body (bounds-checked).
    auto readIntBin = [&](int kind, int size) -> long long {
        if (cur + size > end) { cur = end; return 0; }
        unsigned long long u = 0;
        for (int i = 0; i < size; ++i)
            u |= static_cast<unsigned long long>(static_cast<unsigned char>(cur[i])) << (8 * i);
        cur += size;
        if (kind == 0 && size < 8) {  // signed: sign-extend
            unsigned long long sign = 1ULL << (size * 8 - 1);
            if (u & sign) return static_cast<long long>(u | (~0ULL << (size * 8)));
        }
        return static_cast<long long>(u);
    };
    auto readFloatBin = [&](int size) -> double {
        if (cur + size > end) { cur = end; return 0; }
        double r;
        if (size == 4) { float v; std::memcpy(&v, cur, 4); r = v; }
        else           { double v; std::memcpy(&v, cur, 8); r = v; }
        cur += size;
        return r;
    };
    auto readVal = [&](const Prop& p) -> double {
        return p.kind == 2 ? readFloatBin(p.size)
                           : static_cast<double>(readIntBin(p.kind, p.size));
    };
    // ASCII readers: strtod/strtoll straight off the buffer (no stream overhead).
    auto asciiDouble = [&](double& out) -> bool {
        char* e = nullptr;
        out = std::strtod(cur, &e);
        if (e == cur) return false;
        cur = e;
        return true;
    };
    auto asciiLong = [&](long long& out) -> bool {
        char* e = nullptr;
        out = std::strtoll(cur, &e, 10);
        if (e == cur) return false;
        cur = e;
        return true;
    };

    std::vector<V3> pos(vertexCount);
    std::vector<V3> nrm(vertexCount, V3{0, 0, 0});
    std::vector<V3> col(vertexCount, V3{-1, -1, -1});  // -1 => no color
    // Vertices are the bulk of a point cloud; reading them is 0..70% of progress.
    // Sample progress every 16K points so the per-vertex inner loop stays tight.
    for (int v = 0; v < vertexCount; ++v) {
        if ((v & 0x3FFF) == 0)
            report(0.70f * static_cast<float>(v) / static_cast<float>(vertexCount));
        for (size_t i = 0; i < vprops.size(); ++i) {
            double val;
            if (fmt == ASCII) { if (!asciiDouble(val)) return false; }
            else              val = readVal(vprops[i]);
            int ii = static_cast<int>(i);
            float fv = static_cast<float>(val);
            if (ii == ix) pos[v].x = fv;
            else if (ii == iy) pos[v].y = fv;
            else if (ii == iz) pos[v].z = fv;
            else if (ii == inx) nrm[v].x = fv;
            else if (ii == iny) nrm[v].y = fv;
            else if (ii == inz) nrm[v].z = fv;
            else if (ii == ir) col[v].x = fv / 255.0f;
            else if (ii == ig) col[v].y = fv / 255.0f;
            else if (ii == ib) col[v].z = fv / 255.0f;
        }
    }

    // No faces -> a point cloud: emit one (non-indexed) vertex per point, colored
    // by its PLY color if present, else by its normal. spawnMesh draws it as points.
    if (!haveFace || faceCount == 0) {
        outV.resize(vertexCount);
        for (int v = 0; v < vertexCount; ++v) {
            if (col[v].x >= 0.0f) {
                outV[v] = {{pos[v].x, pos[v].y, pos[v].z}, {col[v].x, col[v].y, col[v].z}};
            } else {
                outV[v] = vert(pos[v], normOrUp(nrm[v]));  // normal -> color
            }
        }
        outI.clear();
        if (progress) progress(1.0f);
        return !outV.empty();
    }

    std::vector<uint32_t>  tris;
    std::vector<long long> idx;  // reused per face (most are triangles)
    if (haveFace) {
        tris.reserve(static_cast<size_t>(faceCount) * 3);
        for (int fc = 0; fc < faceCount; ++fc) {
            if ((fc & 0x3FFF) == 0)
                report(0.70f + 0.25f * static_cast<float>(fc) / static_cast<float>(faceCount));
            long long cnt;
            if (fmt == ASCII) { if (!asciiLong(cnt)) break; }
            else              cnt = readIntBin(faceProp.countKind, faceProp.countSize);
            if (cnt < 0 || cnt > 255) break;  // sanity
            idx.resize(static_cast<size_t>(cnt));
            for (long long k = 0; k < cnt; ++k) {
                if (fmt == ASCII) { if (!asciiLong(idx[k])) return false; }
                else              idx[k] = readIntBin(faceProp.kind, faceProp.size);
            }
            for (long long k = 1; k + 1 < cnt; ++k) {  // fan triangulation
                tris.push_back(static_cast<uint32_t>(idx[0]));
                tris.push_back(static_cast<uint32_t>(idx[k]));
                tris.push_back(static_cast<uint32_t>(idx[k + 1]));
            }
        }
    }
    bool ok = buildIndexed(pos, tris, outV, outI);
    if (progress) progress(1.0f);
    return ok;
}

// Dispatches on the file extension (case-insensitive). Returns false on an
// unknown extension or a parse failure. `progress` (optional) is called with a
// fraction in [0,1] as the load proceeds -- safe to drive a background-thread
// atomic that the UI samples.
inline bool loadMeshFile(const std::string& path, std::vector<render::Vertex>& outV,
                         std::vector<uint32_t>& outI, const ProgressFn& progress = {}) {
    std::string ext;
    auto        dot = path.find_last_of('.');
    if (dot != std::string::npos)
        for (size_t i = dot + 1; i < path.size(); ++i)
            ext.push_back(static_cast<char>(std::tolower(path[i])));
    if (ext == "obj") return loadObj(path.c_str(), outV, outI, progress);
    if (ext == "stl") return loadStl(path.c_str(), outV, outI, progress);
    if (ext == "ply") return loadPly(path.c_str(), outV, outI, progress);
    return false;
}

} // namespace orange::meshio
