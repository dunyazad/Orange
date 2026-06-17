#pragma once

#include "orange/render/types.h"

namespace orange::core {

// A glyph in a baked font atlas. Sizes/offsets are in em units (the bake pixel
// height maps to 1.0), so a caller picks an on-screen height `h` and multiplies.
struct Glyph {
    float u0 = 0, v0 = 0, u1 = 0, v1 = 0;  // atlas UVs (v0 = top)
    float xoff = 0, yoff = 0;              // pen -> glyph top-left (yoff: +down)
    float w = 0, h = 0;                    // glyph quad size
    float advance = 0;                     // pen advance
};

// A baked bitmap font: an RGBA atlas (white text, alpha coverage) + per-glyph
// metrics for ASCII 32..126, plus a fully-white texel for solid UI fills so a
// whole panel can be drawn from this one texture.
struct Font {
    render::TextureHandle texture = render::kInvalidTexture;
    float whiteU = 0.0f, whiteV = 0.0f;
    Glyph glyphs[96] = {};  // index = c - 32

    const Glyph& glyph(char c) const {
        int i = (c >= 32 && c < 127) ? c - 32 : 0;
        return glyphs[i];
    }
    float textWidth(const char* s, float h) const {
        float w = 0.0f;
        for (; *s; ++s) w += glyph(*s).advance * h;
        return w;
    }
};

} // namespace orange::core
