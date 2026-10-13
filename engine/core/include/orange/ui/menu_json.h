#pragma once

#include <string>
#include <vector>

#include "orange/ecs/components.h"
#include "orange/ui/node.h"

// Bridge from a declarative ui::Node tree to the menu bar's data model.
//
// Expected tree (see node.h for the JSON shape):
//   menubar
//     menu        label
//       item      label action [shortcut] [children -> submenu flyout]
//       check     label action [shortcut]   (shows a tick when checked)
//       separator
//       modes     expands to one item per registered processing mode
//                 (modes::modeName(i) -> MenuAction::Mode0 + i), grouped by
//                 ModeCategory with separators, exactly like defaultAppMenus()
//
// Every `action` must name a MenuAction enumerator ("OpenFile", "Mode3", ...);
// an unknown name is a hard error so typos surface at load time rather than
// as dead menu entries.

namespace orange::ui {

// MenuAction enumerator name <-> value. "ModeN" (N = 0..23) is parsed numerically.
bool        menuActionFromName(const std::string& name, ecs::MenuAction& out);
const char* menuActionName(ecs::MenuAction a);  // "" for unnamed values

// Convert a "menubar" node into Menu rows for ecs::MenuBar::menus. On failure
// returns false and leaves `out` untouched; `err` names the node / key.
bool menusFromUi(const Node& menubar, std::vector<ecs::Menu>& out, std::string* err);

} // namespace orange::ui
