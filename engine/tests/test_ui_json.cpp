// Tests for the declarative UI loader (orange/ui): JSON -> ui::Node tree, the
// MenuAction name table, and the menubar bridge. With a path argument it also
// loads that menus.json and checks it reproduces defaultAppMenus() exactly, so
// the data file and the C++ fallback cannot silently drift apart.

#include <cstdio>
#include <string>
#include <vector>

#include "orange/core/modes.h"
#include "orange/ecs/components.h"
#include "orange/ecs/systems.h"
#include "orange/ui/menu_json.h"
#include "orange/ui/node.h"

using namespace orange;

static int g_total = 0, g_fail = 0;
#define CHECK(cond)                                                                \
    do {                                                                           \
        ++g_total;                                                                 \
        if (!(cond)) { ++g_fail; std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

static void testParse() {
    std::printf("parse\n");
    const char* src = R"({
        // comments are allowed
        "type": "panel", "id": "root", "anchor": "bottom-right", "x": 10, "y": -4.5,
        "w": 260, "h": "auto", "pad": 6, "gap": 2, "color": "#ff8800", "flag": true, "n": 3,
        "children": [
            { "type": "row", "w": "fill", "children": [
                { "type": "button", "id": "ok", "label": "OK", "action": "Quit", "shortcut": "Esc", "checked": true }
            ]},
            { "type": "separator" }
        ]
    })";
    ui::Node n;
    std::string err;
    const bool ok = ui::parseUiJson(src, n, &err);
    CHECK(ok);
    if (!ok) std::printf("  err: %s\n", err.c_str());
    CHECK(n.type == "panel");
    CHECK(n.id == "root");
    CHECK(n.anchor == ui::Anchor::BottomRight);
    CHECK(n.x == 10.0f && n.y == -4.5f);
    CHECK(n.w.mode == ui::Size::Fixed && n.w.px == 260.0f);
    CHECK(n.h.mode == ui::Size::Auto);
    CHECK(n.pad == 6.0f && n.gap == 2.0f);
    CHECK(n.props.at("color") == "#ff8800");
    CHECK(n.props.at("flag") == "true");
    CHECK(n.props.at("n") == "3");
    CHECK(n.props.count("type") == 0);  // known keys never land in props
    CHECK(n.children.size() == 2);
    CHECK(n.children[0].w.mode == ui::Size::Fill);
    const ui::Node* okBtn = n.find("ok");
    CHECK(okBtn != nullptr);
    if (okBtn) {
        CHECK(okBtn->label == "OK" && okBtn->action == "Quit" && okBtn->shortcut == "Esc");
        CHECK(okBtn->checked);
    }
    CHECK(n.find("missing") == nullptr);
}

static void testParseErrors() {
    std::printf("parse errors\n");
    ui::Node n;
    std::string err;
    CHECK(!ui::parseUiJson("{ not json", n, &err) && !err.empty());
    err.clear();
    CHECK(!ui::parseUiJson(R"({ "label": "no type" })", n, &err) && err.find("type") != std::string::npos);
    err.clear();
    CHECK(!ui::parseUiJson(R"({ "type": "p", "anchor": "middle" })", n, &err) && err.find("anchor") != std::string::npos);
    err.clear();
    CHECK(!ui::parseUiJson(R"({ "type": "p", "w": "wide" })", n, &err) && err.find(".w") != std::string::npos);
    err.clear();
    CHECK(!ui::parseUiJson(R"({ "type": "p", "children": {} })", n, &err) && err.find("children") != std::string::npos);
    err.clear();
    CHECK(!ui::parseUiJson(R"({ "type": "p", "children": [ { "type": "c", "x": "far" } ] })", n, &err) &&
          err.find("children[0].x") != std::string::npos);
    err.clear();
    CHECK(!ui::loadUiFile("definitely/not/here.json", n, &err) && err.find("cannot open") != std::string::npos);
}

static void testActionNames() {
    std::printf("action names\n");
    using A = ecs::MenuAction;
    // Every enumerator in [None, Compare3D] must round-trip through its name.
    for (int i = static_cast<int>(A::None); i <= static_cast<int>(A::Compare3D); ++i) {
        const A a = static_cast<A>(i);
        const char* name = ui::menuActionName(a);
        A back = A::None;
        const bool named = name && *name;
        CHECK(named);
        if (!named) { std::printf("  unnamed MenuAction %d\n", i); continue; }
        CHECK(ui::menuActionFromName(name, back) && back == a);
    }
    A a;
    CHECK(ui::menuActionFromName("Mode0", a) && a == A::Mode0);
    CHECK(ui::menuActionFromName("Mode23", a) && a == A::Mode23);
    CHECK(!ui::menuActionFromName("Mode24", a));
    CHECK(!ui::menuActionFromName("Mode", a));
    CHECK(!ui::menuActionFromName("ModeX", a));
    CHECK(!ui::menuActionFromName("openfile", a));  // case-sensitive
    CHECK(!ui::menuActionFromName("", a));
}

static bool itemsEqual(const std::vector<ecs::MenuItem>& a, const std::vector<ecs::MenuItem>& b,
                       const std::string& where) {
    if (a.size() != b.size()) {
        std::printf("  %s: item count %zu vs %zu\n", where.c_str(), a.size(), b.size());
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i];
        const auto& y = b[i];
        const std::string w = where + "[" + std::to_string(i) + "]";
        if (x.label != y.label || x.shortcut != y.shortcut || x.action != y.action ||
            x.kind != y.kind || x.checked != y.checked) {
            std::printf("  %s: \"%s\"/%s/%d/%d/%d vs \"%s\"/%s/%d/%d/%d\n", w.c_str(),
                        x.label.c_str(), x.shortcut.c_str(), (int)x.action, (int)x.kind, (int)x.checked,
                        y.label.c_str(), y.shortcut.c_str(), (int)y.action, (int)y.kind, (int)y.checked);
            ok = false;
        }
        if (!itemsEqual(x.submenu, y.submenu, w + ".submenu")) ok = false;
    }
    return ok;
}

static void testMenuBridge() {
    std::printf("menu bridge\n");
    const char* src = R"({
        "type": "menubar",
        "children": [
            { "type": "menu", "label": "File", "children": [
                { "type": "item", "label": "Open...", "action": "OpenFile", "shortcut": "Ctrl+O" },
                { "type": "separator" },
                { "type": "check", "label": "Grid", "action": "ToggleGrid" },
                { "type": "item", "label": "Sub", "children": [
                    { "type": "item", "label": "Leaf", "action": "Mode2" }
                ]}
            ]},
            { "type": "menu", "label": "Geo", "children": [ { "type": "modes" } ] }
        ]
    })";
    ui::Node root;
    std::string err;
    CHECK(ui::parseUiJson(src, root, &err));
    std::vector<ecs::Menu> menus;
    const bool ok = ui::menusFromUi(root, menus, &err);
    CHECK(ok);
    if (!ok) { std::printf("  err: %s\n", err.c_str()); return; }
    CHECK(menus.size() == 2);
    CHECK(menus[0].title == "File" && menus[0].items.size() == 4);
    CHECK(menus[0].items[0].action == ecs::MenuAction::OpenFile && menus[0].items[0].kind == ecs::MenuItem::Action);
    CHECK(menus[0].items[1].kind == ecs::MenuItem::Separator);
    CHECK(menus[0].items[2].kind == ecs::MenuItem::Check && menus[0].items[2].action == ecs::MenuAction::ToggleGrid);
    CHECK(menus[0].items[3].submenu.size() == 1 && menus[0].items[3].submenu[0].action == ecs::MenuAction::Mode2);
    // "modes" expands to one Action item per registered mode (plus category
    // separators), capped at the Mode0..Mode23 action range like defaultAppMenus().
    int actions = 0;
    for (const auto& it : menus[1].items) if (it.kind == ecs::MenuItem::Action) ++actions;
    const int slots = static_cast<int>(ecs::MenuAction::Mode23) - static_cast<int>(ecs::MenuAction::Mode0) + 1;
    CHECK(actions == (modes::modeCount() < slots ? modes::modeCount() : slots));
    if (actions > 0) CHECK(menus[1].items[0].action == ecs::MenuAction::Mode0);

    // Error paths: unknown action, wrong root, deep nesting, stray node type.
    std::vector<ecs::Menu> untouched = menus;
    ui::Node bad;
    err.clear();
    CHECK(ui::parseUiJson(R"({"type":"menubar","children":[{"type":"menu","label":"M","children":[{"type":"item","label":"X","action":"Nope"}]}]})", bad, &err));
    CHECK(!ui::menusFromUi(bad, menus, &err) && err.find("Nope") != std::string::npos);
    CHECK(menus.size() == untouched.size());  // out untouched on failure
    err.clear();
    CHECK(ui::parseUiJson(R"({"type":"panel"})", bad, &err));
    CHECK(!ui::menusFromUi(bad, menus, &err) && err.find("menubar") != std::string::npos);
    err.clear();
    CHECK(ui::parseUiJson(R"({"type":"menubar","children":[{"type":"menu","label":"M","children":[{"type":"item","label":"A","children":[{"type":"item","label":"B","children":[{"type":"item","label":"C","action":"Quit"}]}]}]}]})", bad, &err));
    CHECK(!ui::menusFromUi(bad, menus, &err) && err.find("nest") != std::string::npos);
    err.clear();
    CHECK(ui::parseUiJson(R"({"type":"menubar","children":[{"type":"menu","label":"M","children":[{"type":"button","label":"B"}]}]})", bad, &err));
    CHECK(!ui::menusFromUi(bad, menus, &err) && err.find("button") != std::string::npos);
}

static void testMenusFileMatchesBuiltin(const char* path) {
    std::printf("menus.json == defaultAppMenus()  (%s)\n", path);
    ui::Node root;
    std::string err;
    const bool loaded = ui::loadUiFile(path, root, &err);
    CHECK(loaded);
    if (!loaded) { std::printf("  err: %s\n", err.c_str()); return; }
    std::vector<ecs::Menu> fromJson;
    const bool built = ui::menusFromUi(root, fromJson, &err);
    CHECK(built);
    if (!built) { std::printf("  err: %s\n", err.c_str()); return; }
    const std::vector<ecs::Menu> builtin = ecs::defaultAppMenus();
    CHECK(fromJson.size() == builtin.size());
    const size_t n = fromJson.size() < builtin.size() ? fromJson.size() : builtin.size();
    for (size_t i = 0; i < n; ++i) {
        CHECK(fromJson[i].title == builtin[i].title);
        CHECK(itemsEqual(fromJson[i].items, builtin[i].items, fromJson[i].title));
    }
}

int main(int argc, char** argv) {
    testParse();
    testParseErrors();
    testActionNames();
    testMenuBridge();
    if (argc > 1) testMenusFileMatchesBuiltin(argv[1]);
    std::printf("%d checks, %d failed\n", g_total, g_fail);
    return g_fail ? 1 : 0;
}
