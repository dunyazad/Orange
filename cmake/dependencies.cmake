# ---------------------------------------------------------------------------
# Third-party dependencies, fetched at configure time.
#
# Network access is required the first time you configure. Set
# FETCHCONTENT_SOURCE_DIR_SDL3 / _ENTT to point at a local checkout to work
# offline, or install SDL3/EnTT system-wide and the find_package fallbacks
# below will pick them up.
# ---------------------------------------------------------------------------
include(FetchContent)

# --- SDL3 -------------------------------------------------------------------
# Windowing + input + GL context + Vulkan surface, on desktop AND mobile.
find_package(SDL3 QUIET CONFIG)
if(NOT SDL3_FOUND)
    message(STATUS "Orange: fetching SDL3 ...")
    set(SDL_SHARED ON  CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
    set(SDL_TEST   OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-3.2.10
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(SDL3)
endif()

# --- stb (stb_truetype, header-only, no build) ------------------------------
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
    message(STATUS "Orange: fetching stb ...")
    FetchContent_Populate(stb)
endif()
set(STB_INCLUDE_DIR "${stb_SOURCE_DIR}" CACHE INTERNAL "stb include dir")

# --- EnTT (ECS) -------------------------------------------------------------
find_package(EnTT QUIET CONFIG)
if(NOT EnTT_FOUND)
    message(STATUS "Orange: fetching EnTT ...")
    FetchContent_Declare(
        EnTT
        GIT_REPOSITORY https://github.com/skypjack/entt.git
        GIT_TAG        v3.14.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(EnTT)
endif()
