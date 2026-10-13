#include "orange/ui/menu_json.h"

#include <cstdlib>
#include <cstring>

#include "orange/core/modes.h"

namespace orange::ui {

namespace {

using ecs::MenuAction;

struct ActionEntry { const char* name; MenuAction value; };

// Every MenuAction enumerator except the contiguous Mode0..Mode23 block, which
// is parsed numerically below. Keep in sync with the enum in components.h.
#define ORANGE_ACTION(n) {#n, MenuAction::n}
const ActionEntry kActions[] = {
    ORANGE_ACTION(None),
    ORANGE_ACTION(OpenFile), ORANGE_ACTION(Screenshot), ORANGE_ACTION(Quit),
    ORANGE_ACTION(ToggleGrid), ORANGE_ACTION(ResetCamera), ORANGE_ACTION(ToggleUpAxis),
    ORANGE_ACTION(ToggleProjection),
    ORANGE_ACTION(ToggleLighting), ORANGE_ACTION(ToggleVsync), ORANGE_ACTION(ToggleCrossSection),
    ORANGE_ACTION(ColorOriginal), ORANGE_ACTION(ColorHeight), ORANGE_ACTION(ColorPosition),
    ORANGE_ACTION(ColorGray),
    ORANGE_ACTION(DrawNone), ORANGE_ACTION(DrawSolid), ORANGE_ACTION(DrawWireframe),
    ORANGE_ACTION(DrawWireSolid), ORANGE_ACTION(DrawPoint),
    ORANGE_ACTION(SelectAll), ORANGE_ACTION(DeleteSelected), ORANGE_ACTION(ClearSelection),
    ORANGE_ACTION(UnhideAll),
    ORANGE_ACTION(PointSizeUp), ORANGE_ACTION(PointSizeDown),
    ORANGE_ACTION(ModeOff),
    ORANGE_ACTION(CreatePlane), ORANGE_ACTION(CreateBox), ORANGE_ACTION(CreateSphere),
    ORANGE_ACTION(CreateCylinder), ORANGE_ACTION(CreateCone), ORANGE_ACTION(CreateTorus),
    ORANGE_ACTION(CreateDisk), ORANGE_ACTION(CreateCapsule), ORANGE_ACTION(CreateArrow),
    ORANGE_ACTION(SpatialNone), ORANGE_ACTION(SpatialBVH), ORANGE_ACTION(SpatialOctree),
    ORANGE_ACTION(SpatialKDTree), ORANGE_ACTION(SpatialGrid), ORANGE_ACTION(SpatialLoose),
    ORANGE_ACTION(SpatialBSP), ORANGE_ACTION(SpatialRTree), ORANGE_ACTION(SpatialBall),
    ORANGE_ACTION(PoissonDialogToggle), ORANGE_ACTION(FontSizeDialogToggle),
    ORANGE_ACTION(PipelineOcclusalEstimate), ORANGE_ACTION(PipelineOcclusalWholePCA),
    ORANGE_ACTION(PipelineOcclusalFindCusp), ORANGE_ACTION(PipelineOcclusalPlaneFromCusps),
    ORANGE_ACTION(PipelineOcclusal2DRender),
    ORANGE_ACTION(PipelineSegment2DClassical), ORANGE_ACTION(PipelineSegment2DSAM),
    ORANGE_ACTION(PipelineSegment3D),
    ORANGE_ACTION(Undo), ORANGE_ACTION(Redo),
    ORANGE_ACTION(PipelineDialogToggle),
    ORANGE_ACTION(SaveFile), ORANGE_ACTION(SaveFileAs),
    ORANGE_ACTION(Compare3D),
};
#undef ORANGE_ACTION

constexpr int kModeSlots = static_cast<int>(MenuAction::Mode23) - static_cast<int>(MenuAction::Mode0) + 1;

void fail(std::string* err, const std::string& where, const std::string& msg) {
    if (err) *err = where + ": " + msg;
}

std::string nodeDesc(const Node& n) {
    std::string s = n.type;
    if (!n.label.empty()) s += " \"" + n.label + "\"";
    return s;
}

bool resolveAction(const Node& n, MenuAction& out, std::string* err) {
    if (n.action.empty()) { out = MenuAction::None; return true; }
    if (menuActionFromName(n.action, out)) return true;
    fail(err, nodeDesc(n), "unknown action \"" + n.action + "\"");
    return false;
}

bool buildItems(const Node& parent, std::vector<ecs::MenuItem>& out, int depth, std::string* err);

bool buildItem(const Node& n, std::vector<ecs::MenuItem>& out, int depth, std::string* err) {
    using K = ecs::MenuItem;
    if (n.type == "separator") {
        out.push_back(ecs::MenuItem{"", "", MenuAction::None, K::Separator, false});
        return true;
    }
    if (n.type == "modes") {
        // Generated from the modes registry, grouped by category (same as
        // defaultAppMenus()). Adding a mode in modes.cpp makes it appear here.
        modes::ModeCategory prev = modes::modeCategory(0);
        bool first = true;
        for (int i = 0; i < modes::modeCount() && i < kModeSlots; ++i) {
            modes::ModeCategory cat = modes::modeCategory(i);
            if (!first && cat != prev)
                out.push_back(ecs::MenuItem{"", "", MenuAction::None, K::Separator, false});
            first = false;
            prev  = cat;
            const auto a = static_cast<MenuAction>(static_cast<int>(MenuAction::Mode0) + i);
            out.push_back(ecs::MenuItem{modes::modeName(i), "", a, K::Action, false});
        }
        return true;
    }
    if (n.type == "item" || n.type == "check") {
        if (n.label.empty()) { fail(err, nodeDesc(n), "menu item needs a \"label\""); return false; }
        MenuAction a;
        if (!resolveAction(n, a, err)) return false;
        ecs::MenuItem item{n.label, n.shortcut, a, n.type == "check" ? K::Check : K::Action, n.checked};
        if (!n.children.empty()) {
            if (depth >= 1) { fail(err, nodeDesc(n), "submenus nest one level only"); return false; }
            if (!buildItems(n, item.submenu, depth + 1, err)) return false;
        }
        out.push_back(std::move(item));
        return true;
    }
    fail(err, nodeDesc(n), "unexpected node type inside a menu (want item/check/separator/modes)");
    return false;
}

bool buildItems(const Node& parent, std::vector<ecs::MenuItem>& out, int depth, std::string* err) {
    for (const Node& c : parent.children)
        if (!buildItem(c, out, depth, err)) return false;
    return true;
}

} // namespace

bool menuActionFromName(const std::string& name, MenuAction& out) {
    for (const auto& e : kActions)
        if (name == e.name) { out = e.value; return true; }
    // "ModeN"
    if (name.size() > 4 && name.compare(0, 4, "Mode") == 0) {
        const char* digits = name.c_str() + 4;
        char* end = nullptr;
        const long i = std::strtol(digits, &end, 10);
        if (end && *end == '\0' && i >= 0 && i < kModeSlots) {
            out = static_cast<MenuAction>(static_cast<int>(MenuAction::Mode0) + static_cast<int>(i));
            return true;
        }
    }
    return false;
}

const char* menuActionName(MenuAction a) {
    for (const auto& e : kActions)
        if (e.value == a) return e.name;
    const int i = static_cast<int>(a) - static_cast<int>(MenuAction::Mode0);
    if (i >= 0 && i < kModeSlots) {
        static const char* kModeNames[kModeSlots] = {
            "Mode0",  "Mode1",  "Mode2",  "Mode3",  "Mode4",  "Mode5",  "Mode6",  "Mode7",
            "Mode8",  "Mode9",  "Mode10", "Mode11", "Mode12", "Mode13", "Mode14", "Mode15",
            "Mode16", "Mode17", "Mode18", "Mode19", "Mode20", "Mode21", "Mode22", "Mode23",
        };
        return kModeNames[i];
    }
    return "";
}

bool menusFromUi(const Node& menubar, std::vector<ecs::Menu>& out, std::string* err) {
    if (menubar.type != "menubar") {
        fail(err, nodeDesc(menubar), "root must be of type \"menubar\"");
        return false;
    }
    std::vector<ecs::Menu> menus;
    menus.reserve(menubar.children.size());
    for (const Node& m : menubar.children) {
        if (m.type != "menu") { fail(err, nodeDesc(m), "menubar children must be \"menu\""); return false; }
        if (m.label.empty()) { fail(err, nodeDesc(m), "menu needs a \"label\""); return false; }
        ecs::Menu menu{m.label, {}};
        if (!buildItems(m, menu.items, 0, err)) return false;
        menus.push_back(std::move(menu));
    }
    out = std::move(menus);
    return true;
}

} // namespace orange::ui
