#pragma once

#include <cstddef>

// Minimal, self-contained OpenGL 3.3 core loader.
//
// We deliberately avoid glad/glew: only the handful of entry points the
// renderer actually uses are declared here and resolved via
// SDL_GL_GetProcAddress at runtime. No system <GL/gl.h> is included, so there
// are no header/loader conflicts.

// --- GL base types ----------------------------------------------------------
typedef unsigned int   GLenum;
typedef unsigned char  GLboolean;
typedef unsigned int   GLbitfield;
typedef int            GLint;
typedef int            GLsizei;
typedef unsigned int   GLuint;
typedef float          GLfloat;
typedef char           GLchar;
typedef std::ptrdiff_t GLsizeiptr;
typedef std::ptrdiff_t GLintptr;
typedef void           GLvoid;

// --- Constants we use -------------------------------------------------------
#define GL_FALSE                0
#define GL_TRUE                 1
#define GL_POINTS               0x0000
#define GL_TRIANGLES            0x0004
#define GL_FLOAT                0x1406
#define GL_UNSIGNED_INT         0x1405
#define GL_ARRAY_BUFFER         0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_UNIFORM_BUFFER       0x8A11
#define GL_STATIC_DRAW          0x88E4
#define GL_DYNAMIC_DRAW         0x88E8
#define GL_COLOR_BUFFER_BIT     0x00004000
#define GL_DEPTH_BUFFER_BIT     0x00000100
#define GL_VERTEX_SHADER        0x8B31
#define GL_FRAGMENT_SHADER      0x8B30
#define GL_COMPILE_STATUS       0x8B81
#define GL_LINK_STATUS          0x8B82
#define GL_DEPTH_TEST           0x0B71
#define GL_CULL_FACE            0x0B44
#define GL_STENCIL_TEST         0x0B90
#define GL_STENCIL_BUFFER_BIT   0x00000400
#define GL_KEEP                 0x1E00
#define GL_REPLACE              0x1E01
#define GL_ALWAYS               0x0207
#define GL_NOTEQUAL             0x0205
#define GL_LESS                 0x0201
#define GL_SCISSOR_TEST         0x0C11
#define GL_TEXTURE_2D           0x0DE1
#define GL_RGBA                 0x1908
#define GL_RGBA8                0x8058
#define GL_UNSIGNED_BYTE        0x1401
#define GL_TEXTURE_MAG_FILTER   0x2800
#define GL_TEXTURE_MIN_FILTER   0x2801
#define GL_TEXTURE_WRAP_S       0x2802
#define GL_TEXTURE_WRAP_T       0x2803
#define GL_LINEAR               0x2601
#define GL_CLAMP_TO_EDGE        0x812F
#define GL_TEXTURE0             0x84C0
#define GL_BLEND                0x0BE2
#define GL_SRC_ALPHA            0x0302
#define GL_ONE_MINUS_SRC_ALPHA  0x0303
#define GL_FRONT                0x0404
#define GL_BACK                 0x0405
#define GL_FRONT_AND_BACK       0x0408
#define GL_POINT                0x1B00
#define GL_LINE                 0x1B01
#define GL_FILL                 0x1B02
#define GL_POLYGON_OFFSET_FILL  0x8037
#define GL_PROGRAM_POINT_SIZE   0x8642
#define GL_PACK_ALIGNMENT       0x0D05

// Windows GL calls use __stdcall; everything else is cdecl.
#if defined(_WIN32)
    #define ORANGE_GLAPI __stdcall
#else
    #define ORANGE_GLAPI
#endif

// --- Function pointer typedefs ---------------------------------------------
typedef void   (ORANGE_GLAPI* PFN_glGenVertexArrays)(GLsizei, GLuint*);
typedef void   (ORANGE_GLAPI* PFN_glBindVertexArray)(GLuint);
typedef void   (ORANGE_GLAPI* PFN_glDeleteVertexArrays)(GLsizei, const GLuint*);
typedef void   (ORANGE_GLAPI* PFN_glGenBuffers)(GLsizei, GLuint*);
typedef void   (ORANGE_GLAPI* PFN_glBindBuffer)(GLenum, GLuint);
typedef void   (ORANGE_GLAPI* PFN_glBufferData)(GLenum, GLsizeiptr, const void*, GLenum);
typedef void   (ORANGE_GLAPI* PFN_glBufferSubData)(GLenum, GLintptr, GLsizeiptr, const void*);
typedef void   (ORANGE_GLAPI* PFN_glDeleteBuffers)(GLsizei, const GLuint*);
typedef void   (ORANGE_GLAPI* PFN_glEnableVertexAttribArray)(GLuint);
typedef void   (ORANGE_GLAPI* PFN_glVertexAttribPointer)(GLuint, GLint, GLenum, GLboolean, GLsizei, const void*);
typedef GLuint (ORANGE_GLAPI* PFN_glCreateShader)(GLenum);
typedef void   (ORANGE_GLAPI* PFN_glShaderSource)(GLuint, GLsizei, const GLchar* const*, const GLint*);
typedef void   (ORANGE_GLAPI* PFN_glCompileShader)(GLuint);
typedef void   (ORANGE_GLAPI* PFN_glGetShaderiv)(GLuint, GLenum, GLint*);
typedef void   (ORANGE_GLAPI* PFN_glGetShaderInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void   (ORANGE_GLAPI* PFN_glDeleteShader)(GLuint);
typedef GLuint (ORANGE_GLAPI* PFN_glCreateProgram)(void);
typedef void   (ORANGE_GLAPI* PFN_glAttachShader)(GLuint, GLuint);
typedef void   (ORANGE_GLAPI* PFN_glLinkProgram)(GLuint);
typedef void   (ORANGE_GLAPI* PFN_glGetProgramiv)(GLuint, GLenum, GLint*);
typedef void   (ORANGE_GLAPI* PFN_glGetProgramInfoLog)(GLuint, GLsizei, GLsizei*, GLchar*);
typedef void   (ORANGE_GLAPI* PFN_glUseProgram)(GLuint);
typedef void   (ORANGE_GLAPI* PFN_glDeleteProgram)(GLuint);
typedef GLint  (ORANGE_GLAPI* PFN_glGetUniformLocation)(GLuint, const GLchar*);
typedef void   (ORANGE_GLAPI* PFN_glUniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat*);
typedef void   (ORANGE_GLAPI* PFN_glClearColor)(GLfloat, GLfloat, GLfloat, GLfloat);
typedef void   (ORANGE_GLAPI* PFN_glClear)(GLbitfield);
typedef void   (ORANGE_GLAPI* PFN_glViewport)(GLint, GLint, GLsizei, GLsizei);
typedef void   (ORANGE_GLAPI* PFN_glScissor)(GLint, GLint, GLsizei, GLsizei);
typedef void   (ORANGE_GLAPI* PFN_glEnable)(GLenum);
typedef void   (ORANGE_GLAPI* PFN_glDisable)(GLenum);
typedef void   (ORANGE_GLAPI* PFN_glDepthFunc)(GLenum);
typedef void   (ORANGE_GLAPI* PFN_glDrawElements)(GLenum, GLsizei, GLenum, const void*);
typedef void   (ORANGE_GLAPI* PFN_glDrawArrays)(GLenum, GLint, GLsizei);
typedef void   (ORANGE_GLAPI* PFN_glGenTextures)(GLsizei, GLuint*);
typedef void   (ORANGE_GLAPI* PFN_glBindTexture)(GLenum, GLuint);
typedef void   (ORANGE_GLAPI* PFN_glTexImage2D)(GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*);
typedef void   (ORANGE_GLAPI* PFN_glTexParameteri)(GLenum, GLenum, GLint);
typedef void   (ORANGE_GLAPI* PFN_glActiveTexture)(GLenum);
typedef void   (ORANGE_GLAPI* PFN_glDeleteTextures)(GLsizei, const GLuint*);
typedef void   (ORANGE_GLAPI* PFN_glUniform1i)(GLint, GLint);
typedef void   (ORANGE_GLAPI* PFN_glUniform1f)(GLint, GLfloat);
typedef void   (ORANGE_GLAPI* PFN_glUniform4f)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
typedef void   (ORANGE_GLAPI* PFN_glUniform3fv)(GLint, GLsizei, const GLfloat*);
typedef void   (ORANGE_GLAPI* PFN_glBlendFunc)(GLenum, GLenum);
typedef void   (ORANGE_GLAPI* PFN_glReadPixels)(GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*);
typedef void   (ORANGE_GLAPI* PFN_glReadBuffer)(GLenum);
typedef void   (ORANGE_GLAPI* PFN_glPixelStorei)(GLenum, GLint);
typedef void   (ORANGE_GLAPI* PFN_glPolygonMode)(GLenum, GLenum);
typedef void   (ORANGE_GLAPI* PFN_glPointSize)(GLfloat);
typedef void   (ORANGE_GLAPI* PFN_glPolygonOffset)(GLfloat, GLfloat);
typedef void   (ORANGE_GLAPI* PFN_glCullFace)(GLenum);
typedef void   (ORANGE_GLAPI* PFN_glColorMask)(GLboolean, GLboolean, GLboolean, GLboolean);
typedef void   (ORANGE_GLAPI* PFN_glStencilFunc)(GLenum, GLint, GLuint);
typedef void   (ORANGE_GLAPI* PFN_glStencilOp)(GLenum, GLenum, GLenum);
typedef void   (ORANGE_GLAPI* PFN_glStencilMask)(GLuint);

// --- Function pointers (defined in gl_loader.cpp) --------------------------
extern PFN_glGenVertexArrays         glGenVertexArrays;
extern PFN_glBindVertexArray         glBindVertexArray;
extern PFN_glDeleteVertexArrays      glDeleteVertexArrays;
extern PFN_glGenBuffers              glGenBuffers;
extern PFN_glBindBuffer              glBindBuffer;
extern PFN_glBufferData              glBufferData;
extern PFN_glBufferSubData           glBufferSubData;
extern PFN_glDeleteBuffers           glDeleteBuffers;
extern PFN_glEnableVertexAttribArray glEnableVertexAttribArray;
extern PFN_glVertexAttribPointer     glVertexAttribPointer;
extern PFN_glCreateShader            glCreateShader;
extern PFN_glShaderSource            glShaderSource;
extern PFN_glCompileShader           glCompileShader;
extern PFN_glGetShaderiv             glGetShaderiv;
extern PFN_glGetShaderInfoLog        glGetShaderInfoLog;
extern PFN_glDeleteShader            glDeleteShader;
extern PFN_glCreateProgram           glCreateProgram;
extern PFN_glAttachShader            glAttachShader;
extern PFN_glLinkProgram             glLinkProgram;
extern PFN_glGetProgramiv            glGetProgramiv;
extern PFN_glGetProgramInfoLog       glGetProgramInfoLog;
extern PFN_glUseProgram              glUseProgram;
extern PFN_glDeleteProgram           glDeleteProgram;
extern PFN_glGetUniformLocation      glGetUniformLocation;
extern PFN_glUniformMatrix4fv        glUniformMatrix4fv;
extern PFN_glClearColor              glClearColor;
extern PFN_glClear                   glClear;
extern PFN_glViewport                glViewport;
extern PFN_glScissor                 glScissor;
extern PFN_glEnable                  glEnable;
extern PFN_glDisable                 glDisable;
extern PFN_glDepthFunc               glDepthFunc;
extern PFN_glDrawElements            glDrawElements;
extern PFN_glDrawArrays              glDrawArrays;
extern PFN_glGenTextures             glGenTextures;
extern PFN_glBindTexture             glBindTexture;
extern PFN_glTexImage2D              glTexImage2D;
extern PFN_glTexParameteri           glTexParameteri;
extern PFN_glActiveTexture           glActiveTexture;
extern PFN_glDeleteTextures          glDeleteTextures;
extern PFN_glUniform1i               glUniform1i;
extern PFN_glUniform1f               glUniform1f;
extern PFN_glUniform4f               glUniform4f;
extern PFN_glUniform3fv              glUniform3fv;
extern PFN_glBlendFunc               glBlendFunc;
extern PFN_glReadPixels              glReadPixels;
extern PFN_glReadBuffer              glReadBuffer;
extern PFN_glPixelStorei             glPixelStorei;
extern PFN_glPolygonMode             glPolygonMode;
extern PFN_glPointSize               glPointSize;
extern PFN_glPolygonOffset           glPolygonOffset;
extern PFN_glCullFace                glCullFace;
extern PFN_glColorMask               glColorMask;
extern PFN_glStencilFunc             glStencilFunc;
extern PFN_glStencilOp               glStencilOp;
extern PFN_glStencilMask             glStencilMask;

namespace orange::gl {
// Resolve all entry points. Requires a current GL context. Returns false if
// any required function is missing.
bool loadGLFunctions();
} // namespace orange::gl
