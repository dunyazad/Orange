#include "orange/core/ui_layout.h"

#include <fstream>
#include <string>

#include "orange/ecs/components.h"

// Tiny line-based key/value store for the draggable widgets' last positions:
//   fps  <relX> <relY>
//   tree <relX> <relY> <expanded0> <expanded1>
// Positions are viewport fractions (the same units the widgets store); the input
// systems clamp them on screen, so a stale value from a different window size is
// harmless.

namespace orange::core {

void saveWidgetLayout(const entt::registry& world, const std::string& path) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return;

    auto fps = world.view<const ecs::FpsWidget>();
    for (auto e : fps) {
        const auto& w = fps.get<const ecs::FpsWidget>(e);
        f << "fps " << w.relX << ' ' << w.relY << '\n';
        break;
    }
    auto tree = world.view<const ecs::TreeView>();
    for (auto e : tree) {
        const auto& w = tree.get<const ecs::TreeView>(e);
        f << "tree " << w.relX << ' ' << w.relY << ' ' << (w.expanded[0] ? 1 : 0) << ' '
          << (w.expanded[1] ? 1 : 0) << '\n';
        break;
    }
}

void loadWidgetLayout(entt::registry& world, const std::string& path) {
    std::ifstream f(path);
    if (!f) return;

    std::string tag;
    while (f >> tag) {
        if (tag == "fps") {
            float x = 0.0f, y = 0.0f;
            if (!(f >> x >> y)) break;
            auto v = world.view<ecs::FpsWidget>();
            for (auto e : v) { auto& w = v.get<ecs::FpsWidget>(e); w.relX = x; w.relY = y; break; }
        } else if (tag == "tree") {
            float x = 0.0f, y = 0.0f;
            int e0 = 1, e1 = 1;
            if (!(f >> x >> y >> e0 >> e1)) break;
            auto v = world.view<ecs::TreeView>();
            for (auto e : v) {
                auto& w = v.get<ecs::TreeView>(e);
                w.relX = x; w.relY = y;
                w.expanded[0] = e0 != 0;
                w.expanded[1] = e1 != 0;
                break;
            }
        } else {
            std::getline(f, tag);  // skip the rest of an unknown line
        }
    }
}

} // namespace orange::core
