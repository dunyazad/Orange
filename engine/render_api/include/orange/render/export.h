#pragma once

// Symbol visibility helpers shared by the engine and render plugins.
//
// A render plugin is a shared library that exports a small set of C entry
// points (see plugin_abi.h). Those entry points are always compiled with
// ORANGE_PLUGIN_EXPORT so the host can resolve them by name at runtime.

#if defined(_WIN32)
    #define ORANGE_PLUGIN_EXPORT __declspec(dllexport)
    #define ORANGE_PLUGIN_IMPORT __declspec(dllimport)
#else
    #define ORANGE_PLUGIN_EXPORT __attribute__((visibility("default")))
    #define ORANGE_PLUGIN_IMPORT
#endif
