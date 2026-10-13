#pragma once

#include <map>
#include <string>
#include <vector>

// Declarative 2D UI description loaded from JSON.
//
// A UI file is a tree of Nodes. Each node has a `type` string (what widget it
// is), optional `id`, text (`label`/`shortcut`), a command binding (`action` =
// a MenuAction name, `bind` = a property path, both plain strings so the JSON
// stays free of callbacks), layout hints (anchor / x / y / w / h / pad / gap)
// and `children`. Keys the parser does not know are kept verbatim in `props`
// so a widget type can carry its own extras without touching the parser.
//
// Step 1 (this file) only defines the tree + the JSON parser. The menu bar is
// the first consumer (menu_json.h turns a "menubar" tree into ecs::Menu rows);
// the generic layout / input / render systems for panels come next.
//
// JSON shape:
//   {
//     "type": "menubar",
//     "children": [
//       { "type": "menu", "label": "File", "children": [
//         { "type": "item", "label": "Open...", "action": "OpenFile", "shortcut": "Ctrl+O" },
//         { "type": "separator" },
//         { "type": "check", "label": "Ground Grid", "action": "ToggleGrid" }
//       ]}
//     ]
//   }
//
// Sizes: a number means fixed pixels; the strings "fill" (take the remaining
// space) and "auto" (fit content, the default) are the two symbolic modes.

namespace orange::ui {

struct Size {
    enum Mode { Auto, Fixed, Fill };
    Mode  mode = Auto;
    float px   = 0.0f;  // valid when mode == Fixed
};

enum class Anchor {
    TopLeft, Top, TopRight,
    Left, Center, Right,
    BottomLeft, Bottom, BottomRight,
};

struct Node {
    std::string id;        // optional unique name (used by bindings / lookups)
    std::string type;      // widget kind: "menubar", "menu", "item", "check", "separator", "modes", ...
    std::string label;     // display text
    std::string shortcut;  // right-aligned hint, e.g. "Ctrl+O" (menu items)
    std::string action;    // ecs::MenuAction enumerator name, e.g. "OpenFile"
    std::string bind;      // property path, e.g. "crossSection.pos" (reserved for later steps)

    Anchor anchor = Anchor::TopLeft;
    float  x = 0.0f, y = 0.0f;  // offset from the anchor, px
    Size   w, h;
    float  pad = 0.0f, gap = 0.0f;
    bool   checked = false;

    // Unrecognised scalar keys (string / number / bool), stringified.
    std::map<std::string, std::string> props;
    std::vector<Node> children;

    // First descendant (depth-first, including this node) whose id matches, or
    // nullptr.
    const Node* find(const std::string& id) const;
    Node*       find(const std::string& id);
};

// Parse a UI tree from JSON text. On failure returns false and, if `err` is
// non-null, stores a message with the offending key / position.
bool parseUiJson(const std::string& text, Node& out, std::string* err);

// Read + parse a UI file. Failure (missing file, bad JSON, unknown enum value)
// returns false with a message in `err`.
bool loadUiFile(const std::string& path, Node& out, std::string* err);

// String <-> enum helpers (used by the parser; exposed for tests / tooling).
bool        anchorFromName(const std::string& name, Anchor& out);
const char* anchorName(Anchor a);

} // namespace orange::ui
