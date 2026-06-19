#include "gl_loader.h"

#include <SDL3/SDL.h>

// Definitions of the function pointers declared in gl_loader.h.
PFN_glGenVertexArrays         glGenVertexArrays         = nullptr;
PFN_glBindVertexArray         glBindVertexArray         = nullptr;
PFN_glDeleteVertexArrays      glDeleteVertexArrays      = nullptr;
PFN_glGenBuffers              glGenBuffers              = nullptr;
PFN_glBindBuffer              glBindBuffer              = nullptr;
PFN_glBufferData              glBufferData              = nullptr;
PFN_glBufferSubData           glBufferSubData           = nullptr;
PFN_glDeleteBuffers           glDeleteBuffers           = nullptr;
PFN_glEnableVertexAttribArray glEnableVertexAttribArray = nullptr;
PFN_glVertexAttribPointer     glVertexAttribPointer     = nullptr;
PFN_glCreateShader            glCreateShader            = nullptr;
PFN_glShaderSource            glShaderSource            = nullptr;
PFN_glCompileShader           glCompileShader           = nullptr;
PFN_glGetShaderiv             glGetShaderiv             = nullptr;
PFN_glGetShaderInfoLog        glGetShaderInfoLog        = nullptr;
PFN_glDeleteShader            glDeleteShader            = nullptr;
PFN_glCreateProgram           glCreateProgram           = nullptr;
PFN_glAttachShader            glAttachShader            = nullptr;
PFN_glLinkProgram             glLinkProgram             = nullptr;
PFN_glGetProgramiv            glGetProgramiv            = nullptr;
PFN_glGetProgramInfoLog       glGetProgramInfoLog       = nullptr;
PFN_glUseProgram              glUseProgram              = nullptr;
PFN_glDeleteProgram           glDeleteProgram           = nullptr;
PFN_glGetUniformLocation      glGetUniformLocation      = nullptr;
PFN_glUniformMatrix4fv        glUniformMatrix4fv        = nullptr;
PFN_glClearColor              glClearColor              = nullptr;
PFN_glClear                   glClear                   = nullptr;
PFN_glViewport                glViewport                = nullptr;
PFN_glScissor                 glScissor                 = nullptr;
PFN_glEnable                  glEnable                  = nullptr;
PFN_glDisable                 glDisable                 = nullptr;
PFN_glDepthFunc               glDepthFunc               = nullptr;
PFN_glDrawElements            glDrawElements            = nullptr;
PFN_glDrawArrays              glDrawArrays              = nullptr;
PFN_glGenTextures             glGenTextures             = nullptr;
PFN_glBindTexture             glBindTexture             = nullptr;
PFN_glTexImage2D              glTexImage2D              = nullptr;
PFN_glTexParameteri           glTexParameteri           = nullptr;
PFN_glActiveTexture           glActiveTexture           = nullptr;
PFN_glDeleteTextures          glDeleteTextures          = nullptr;
PFN_glUniform1i               glUniform1i               = nullptr;
PFN_glUniform4f               glUniform4f               = nullptr;
PFN_glBlendFunc               glBlendFunc               = nullptr;
PFN_glReadPixels              glReadPixels              = nullptr;
PFN_glReadBuffer              glReadBuffer              = nullptr;
PFN_glPixelStorei             glPixelStorei             = nullptr;
PFN_glPolygonMode             glPolygonMode             = nullptr;
PFN_glPointSize               glPointSize               = nullptr;
PFN_glPolygonOffset           glPolygonOffset           = nullptr;

namespace orange::gl {

namespace {
bool g_ok = true;

template <typename T>
void load(T& fn, const char* name) {
    fn = reinterpret_cast<T>(SDL_GL_GetProcAddress(name));
    if (!fn) {
        SDL_Log("gl_loader: missing GL function '%s'", name);
        g_ok = false;
    }
}
} // namespace

#define ORANGE_LOAD(name) load(name, #name)

bool loadGLFunctions() {
    g_ok = true;
    ORANGE_LOAD(glGenVertexArrays);
    ORANGE_LOAD(glBindVertexArray);
    ORANGE_LOAD(glDeleteVertexArrays);
    ORANGE_LOAD(glGenBuffers);
    ORANGE_LOAD(glBindBuffer);
    ORANGE_LOAD(glBufferData);
    ORANGE_LOAD(glBufferSubData);
    ORANGE_LOAD(glDeleteBuffers);
    ORANGE_LOAD(glEnableVertexAttribArray);
    ORANGE_LOAD(glVertexAttribPointer);
    ORANGE_LOAD(glCreateShader);
    ORANGE_LOAD(glShaderSource);
    ORANGE_LOAD(glCompileShader);
    ORANGE_LOAD(glGetShaderiv);
    ORANGE_LOAD(glGetShaderInfoLog);
    ORANGE_LOAD(glDeleteShader);
    ORANGE_LOAD(glCreateProgram);
    ORANGE_LOAD(glAttachShader);
    ORANGE_LOAD(glLinkProgram);
    ORANGE_LOAD(glGetProgramiv);
    ORANGE_LOAD(glGetProgramInfoLog);
    ORANGE_LOAD(glUseProgram);
    ORANGE_LOAD(glDeleteProgram);
    ORANGE_LOAD(glGetUniformLocation);
    ORANGE_LOAD(glUniformMatrix4fv);
    ORANGE_LOAD(glClearColor);
    ORANGE_LOAD(glClear);
    ORANGE_LOAD(glViewport);
    ORANGE_LOAD(glScissor);
    ORANGE_LOAD(glEnable);
    ORANGE_LOAD(glDisable);
    ORANGE_LOAD(glDepthFunc);
    ORANGE_LOAD(glDrawElements);
    ORANGE_LOAD(glDrawArrays);
    ORANGE_LOAD(glGenTextures);
    ORANGE_LOAD(glBindTexture);
    ORANGE_LOAD(glTexImage2D);
    ORANGE_LOAD(glTexParameteri);
    ORANGE_LOAD(glActiveTexture);
    ORANGE_LOAD(glDeleteTextures);
    ORANGE_LOAD(glUniform1i);
    ORANGE_LOAD(glUniform4f);
    ORANGE_LOAD(glBlendFunc);
    ORANGE_LOAD(glReadPixels);
    ORANGE_LOAD(glReadBuffer);
    ORANGE_LOAD(glPixelStorei);
    ORANGE_LOAD(glPolygonMode);
    ORANGE_LOAD(glPointSize);
    ORANGE_LOAD(glPolygonOffset);
    return g_ok;
}

#undef ORANGE_LOAD

} // namespace orange::gl
