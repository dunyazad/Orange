#pragma once

namespace orange::core {

// Per-frame input snapshot. The Application fills this from SDL events; systems
// (e.g. the camera manipulator) read it. Deltas are accumulated within a frame
// and reset by newFrame() at the start of the next.
struct Input {
    float mousePosX = 0.0f, mousePosY = 0.0f;  // window-space cursor position
    float mouseDeltaX = 0.0f, mouseDeltaY = 0.0f;  // movement since last frame
    float wheel = 0.0f;                            // scroll delta this frame

    bool buttonLeft   = false;
    bool buttonRight  = false;
    bool buttonMiddle = false;

    bool leftClicked = false;  // left button went down this frame (edge)
    bool captured    = false;  // a UI widget consumed the mouse this frame
    bool shift       = false;  // a Shift key is held this frame (level state)
    bool ctrl        = false;  // a Ctrl key is held this frame (level state)

    // Clear accumulators that are only meaningful for a single frame.
    void newFrame() {
        mouseDeltaX = 0.0f;
        mouseDeltaY = 0.0f;
        wheel       = 0.0f;
        leftClicked = false;
        captured    = false;
    }
};

} // namespace orange::core
