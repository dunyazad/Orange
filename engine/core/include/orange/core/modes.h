#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <Eigen/Core>

#include "orange/core/debug_draw.h"

// Selectable point-cloud processing "modes" — the Orange equivalent of the
// Hydrogen apps. Each mode takes an input point cloud and emits a visualization
// (colored points, boxes, a reconstructed mesh) into a DebugDraw. The host
// cycles the active mode with a key; processingModeSystem runs the active mode
// when it changes and re-emits the cached result each frame.
//
// The algorithms (Euclidean clustering, voxel morphology, SDF denoise, TSDF
// surface reconstruction) are ported from Elements/Helium Hydrogen apps,
// decoupled from CUDA/Helium and built on Orange's geometry toolkit.

namespace orange::modes {

// Input data a mode runs on (populated by the app into the registry ctx).
struct ModeInput {
    std::vector<Eigen::Vector3f> points;
    std::vector<Eigen::Vector3f> normals;  // optional (used by reconstruction)
    std::vector<float> params;  // per-mode tunables (see modeParam); empty => defaults
    // Optional candidate mask (size == points.size(), 1 = eligible), fed by the
    // Pipeline's condition pin: filters that support it (modeSupportsCandidates)
    // still build their reference statistics from ALL points but only ever
    // remove/project the flagged candidates. Empty = every point eligible.
    std::vector<uint8_t> candidates;
};

// Active-mode selector, stored in the registry ctx. The host bumps `generation`
// whenever it changes `index` so the system knows to recompute. `index < 0` means
// NO mode is active -- the default, so merely selecting an entity does not run a
// processing operator (it must be turned on from the Geometry menu / M key).
struct ModeState {
    int      index      = -1;
    uint64_t generation = 1;
};

// Operator family, used to group the modes in the menu. Filters mark points
// keep/drop; Analyze maps a per-point scalar to a heatmap; Generate emits new
// geometry (a reconstructed mesh / resampled cloud).
enum class ModeCategory { Filter = 0, Analyze, Generate, Transform };

// The registered modes (stable order; index into this list).
int          modeCount();
const char*  modeName(int index);
ModeCategory modeCategory(int index);
const char*  modeCategoryName(ModeCategory c);

// Progress callback: a long-running mode reports its completion fraction in
// [0,1] as it works (throttled, not per-point). May be empty (ignored). Called
// from the worker thread, so the sink must be thread-safe.
using ProgressFn = std::function<void(float)>;

// Run mode `index` on `in`, emitting its visualization into `out`, reporting
// progress through `progress`. Modes that need normals estimate them from the
// points when `in.normals` is empty.
void runMode(int index, const ModeInput& in, debug::DebugDraw& out,
             const ProgressFn& progress = {});

// How the host applies a mode's result. Draw modes emit standalone debug
// geometry (new/moved geometry: reconstruction, smoothing, ICP). Recolor modes
// classify each input point -- the host paints the source cloud's own vertex
// colors in place (restored when the mode ends) instead of overlaying debug
// points. Remove modes delete the flagged points from the source buffer.
// Recolor/Remove fall back to a drawn visualization when the host can't edit
// the source (e.g. a fixed ctx input).
enum class ApplyKind { Draw = 0, Recolor, Remove };
ApplyKind modeApplyKind(int index);

// A tunable parameter of a mode, edited in the ModeParamsDialog. Values reach
// the mode through ModeInput::params (index-aligned with modeParam order);
// "%" parameters are percentages of the input's bounding-box diagonal.
// A name starting with "--" is a section HEADER: the dialog draws it as a
// labeled divider with a CHECKBOX instead of a slider, and its params slot
// holds the section's enable flag (0 = off, 1 = on; clicking toggles). Modes
// that don't read the flag just leave defV there -- the index alignment above
// stays trivial either way.
struct ModeParam {
    const char* name;
    float minV, maxV, defV;
    bool  isInt;
    float step = 0.0f;  // slider/(+/-) increment; 0 => continuous (int params step 1)
};
int       modeParamCount(int index);
ModeParam modeParam(int index, int p);
inline bool modeParamIsHeader(const ModeParam& p) {
    return p.name && p.name[0] == '-' && p.name[1] == '-';
}

// Run a Recolor mode: fills `colors` (one entry per input point; a color with
// x < 0 keeps that point's original color). Extra non-point visualization
// (e.g. cluster boxes) lands in `extras`.
void runModeColors(int index, const ModeInput& in, std::vector<Eigen::Vector3f>& colors,
                   debug::DebugDraw& extras, const ProgressFn& progress = {});

// Legend metadata published by the LAST runModeColors call: the scalar scale a
// heatmap mode's colors saturate at (valid == false when that mode has no
// numeric legend -- e.g. the keep/drop filters). The host polls this after a
// recolor finishes and shows/hides the on-screen color-bar legend. Thread-safe
// (the worker publishes, the main thread polls after completion).
struct ModeLegend {
    bool  valid    = false;
    bool  isSigned = true;
    float range    = 0.0f;  // colors saturate at +-range (signed) or 0..range
    int   bands    = 1;     // quantization steps (1 = smooth ramp)
    // Per-point scalar the colors encode (size == input points): lets the host
    // range-filter the display (legend thumbs) without re-running the mode.
    std::vector<float> scalars;
};
ModeLegend modeLastLegend();

// Run a Remove mode: fills `mask` (one entry per input point, 1 = delete).
void runModeMask(int index, const ModeInput& in, std::vector<uint8_t>& mask,
                 const ProgressFn& progress = {});

// Some modes can FIT the points instead of just classifying them (e.g. PFOR
// projects its outliers onto their local fitted plane). The dialog shows a
// "Fit" button for these; the host replaces the cloud's positions (undoable).
bool modeCanFit(int index);
// Draw modes whose triangle output can be applied as a REAL mesh entity from
// the dialog's Fit button (Reconstruct): the host spawns the cached triangles
// as a pickable/editable/saveable Renderable and deactivates the mode.
bool modeAppliesMesh(int index);
// Fills `fitted` with the new world-space positions (same order/count as
// in.points; non-fitted points keep their input position).
void runModeFit(int index, const ModeInput& in, std::vector<Eigen::Vector3f>& fitted,
                const ProgressFn& progress = {});

// Pipeline-graph execution (the Pipeline Design node canvas): modes that can
// act as a points -> points stage. Filters output only their kept points,
// Smooth outputs the moved points, Bump Remove drops its blobs. Chain stages
// by feeding one stage's output into the next.
bool modeCanTransformPoints(int index);
// True when the mode's Fit action exists AND differs from its points stage
// (e.g. PFOR: points drops outliers, Fit projects them). Such modes get a
// second "<name> Fit" stage on the Pipeline canvas.
bool modeFitIsDistinct(int index);
// Filters that honor ModeInput::candidates (the Pipeline condition pin).
bool modeSupportsCandidates(int index);
void runModePoints(int index, const ModeInput& in, std::vector<Eigen::Vector3f>& outPoints,
                   const ProgressFn& progress = {});
// Stable lookup of a mode by its registered display name (-1 if unknown).
int modeIndexByName(const char* name);

} // namespace orange::modes
