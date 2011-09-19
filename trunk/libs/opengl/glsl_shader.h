#ifndef _GLSL_SHADER_
#define _GLSL_SHADER_

#include <opengl/opengl_utilities.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    GLuint shaderProgram;
    GLuint vertexShaderId;
    GLuint fragmentShaderId;
} glsl_shader;

void glsl_shader_init(glsl_shader* shader, const char* vertex_program_file, const char* fragment_program_file);
void glsl_shader_apply(glsl_shader* shader);
void glsl_shader_release(glsl_shader* shader);

#ifdef __cplusplus
}
#endif

#endif /* _GLSL_SHADER_ */
