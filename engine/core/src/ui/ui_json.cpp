#include "orange/ui/node.h"

#include <fstream>
#include <sstream>

// nlohmann/json is vendored under engine/third_party (header-only) and used
// only from this translation unit: it never appears in a public header, so the
// installed SDK does not ship or depend on it.
#include <nlohmann/json.hpp>

namespace orange::ui {

using json = nlohmann::json;

namespace {

struct AnchorEntry { const char* name; Anchor value; };
const AnchorEntry kAnchors[] = {
    {"top-left",     Anchor::TopLeft},
    {"top",          Anchor::Top},
    {"top-right",    Anchor::TopRight},
    {"left",         Anchor::Left},
    {"center",       Anchor::Center},
    {"right",        Anchor::Right},
    {"bottom-left",  Anchor::BottomLeft},
    {"bottom",       Anchor::Bottom},
    {"bottom-right", Anchor::BottomRight},
};

void fail(std::string* err, const std::string& path, const std::string& msg) {
    if (err) *err = path + ": " + msg;
}

bool parseSize(const json& j, Size& out, const std::string& path, std::string* err) {
    if (j.is_number()) {
        out.mode = Size::Fixed;
        out.px   = j.get<float>();
        return true;
    }
    if (j.is_string()) {
        const std::string s = j.get<std::string>();
        if (s == "fill") { out.mode = Size::Fill;  out.px = 0; return true; }
        if (s == "auto") { out.mode = Size::Auto;  out.px = 0; return true; }
        fail(err, path, "size must be a number, \"fill\" or \"auto\" (got \"" + s + "\")");
        return false;
    }
    fail(err, path, "size must be a number or string");
    return false;
}

bool getString(const json& j, const char* key, std::string& out, const std::string& path, std::string* err) {
    auto it = j.find(key);
    if (it == j.end()) return true;
    if (!it->is_string()) { fail(err, path + "." + key, "expected string"); return false; }
    out = it->get<std::string>();
    return true;
}

bool getFloat(const json& j, const char* key, float& out, const std::string& path, std::string* err) {
    auto it = j.find(key);
    if (it == j.end()) return true;
    if (!it->is_number()) { fail(err, path + "." + key, "expected number"); return false; }
    out = it->get<float>();
    return true;
}

bool getBool(const json& j, const char* key, bool& out, const std::string& path, std::string* err) {
    auto it = j.find(key);
    if (it == j.end()) return true;
    if (!it->is_boolean()) { fail(err, path + "." + key, "expected bool"); return false; }
    out = it->get<bool>();
    return true;
}

bool isKnownKey(const std::string& k) {
    static const char* kKnown[] = {
        "id", "type", "label", "shortcut", "action", "bind",
        "anchor", "x", "y", "w", "h", "pad", "gap", "checked", "children",
    };
    for (const char* n : kKnown)
        if (k == n) return true;
    return false;
}

bool parseNode(const json& j, Node& out, const std::string& path, std::string* err) {
    if (!j.is_object()) { fail(err, path, "node must be an object"); return false; }

    if (!getString(j, "id",       out.id,       path, err)) return false;
    if (!getString(j, "type",     out.type,     path, err)) return false;
    if (!getString(j, "label",    out.label,    path, err)) return false;
    if (!getString(j, "shortcut", out.shortcut, path, err)) return false;
    if (!getString(j, "action",   out.action,   path, err)) return false;
    if (!getString(j, "bind",     out.bind,     path, err)) return false;
    if (out.type.empty()) { fail(err, path, "missing \"type\""); return false; }

    if (auto it = j.find("anchor"); it != j.end()) {
        if (!it->is_string()) { fail(err, path + ".anchor", "expected string"); return false; }
        if (!anchorFromName(it->get<std::string>(), out.anchor)) {
            fail(err, path + ".anchor", "unknown anchor \"" + it->get<std::string>() + "\"");
            return false;
        }
    }
    if (!getFloat(j, "x",   out.x,   path, err)) return false;
    if (!getFloat(j, "y",   out.y,   path, err)) return false;
    if (!getFloat(j, "pad", out.pad, path, err)) return false;
    if (!getFloat(j, "gap", out.gap, path, err)) return false;
    if (auto it = j.find("w"); it != j.end() && !parseSize(*it, out.w, path + ".w", err)) return false;
    if (auto it = j.find("h"); it != j.end() && !parseSize(*it, out.h, path + ".h", err)) return false;
    if (!getBool(j, "checked", out.checked, path, err)) return false;

    // Unknown scalar keys are preserved for widget-specific use.
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (isKnownKey(it.key())) continue;
        const json& v = it.value();
        if (v.is_string())       out.props[it.key()] = v.get<std::string>();
        else if (v.is_boolean()) out.props[it.key()] = v.get<bool>() ? "true" : "false";
        else if (v.is_number())  out.props[it.key()] = v.dump();
        else { fail(err, path + "." + it.key(), "unsupported value type for extra key"); return false; }
    }

    if (auto it = j.find("children"); it != j.end()) {
        if (!it->is_array()) { fail(err, path + ".children", "expected array"); return false; }
        out.children.reserve(it->size());
        size_t i = 0;
        for (const json& c : *it) {
            Node child;
            const std::string cpath = path + ".children[" + std::to_string(i++) + "]";
            if (!parseNode(c, child, cpath, err)) return false;
            out.children.push_back(std::move(child));
        }
    }
    return true;
}

template <class N>
N* findImpl(N& n, const std::string& id) {
    if (n.id == id) return &n;
    for (auto& c : n.children)
        if (N* r = findImpl(c, id)) return r;
    return nullptr;
}

} // namespace

const Node* Node::find(const std::string& id) const { return findImpl(*this, id); }
Node*       Node::find(const std::string& id)       { return findImpl(*this, id); }

bool anchorFromName(const std::string& name, Anchor& out) {
    for (const auto& e : kAnchors)
        if (name == e.name) { out = e.value; return true; }
    return false;
}

const char* anchorName(Anchor a) {
    for (const auto& e : kAnchors)
        if (e.value == a) return e.name;
    return "top-left";
}

bool parseUiJson(const std::string& text, Node& out, std::string* err) {
    json j = json::parse(text, /*cb=*/nullptr, /*allow_exceptions=*/false,
                         /*ignore_comments=*/true);
    if (j.is_discarded()) {
        // Re-parse with exceptions to get the byte position in the message.
        try {
            (void)json::parse(text, nullptr, true, true);
        } catch (const json::parse_error& e) {
            fail(err, "json", e.what());
            return false;
        }
        fail(err, "json", "parse error");
        return false;
    }
    Node tmp;
    if (!parseNode(j, tmp, "root", err)) return false;
    out = std::move(tmp);
    return true;
}

bool loadUiFile(const std::string& path, Node& out, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { fail(err, path, "cannot open file"); return false; }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string e;
    if (!parseUiJson(ss.str(), out, &e)) {
        fail(err, path, e);
        return false;
    }
    return true;
}

} // namespace orange::ui
