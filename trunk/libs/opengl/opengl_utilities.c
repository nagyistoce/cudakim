#include <opengl/opengl_utilities.h>

#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include <common/error.h>

int isSupported(const char* ext) {
    char* str = (char*)glGetString(GL_EXTENSIONS);
    return (strstr((const char *)str, ext) == NULL) ? 0 : 1;
}

void inforce_extention_support(const char* ext) {
    if (!isSupported(ext))
        exit_with_error("OpenGL extension not supported: %s\n", ext);
}

/**
 *  Should never be used in the code, use CHECK_FOR_GL_ERROR(); instead
 */
void CHECK_FOR_GL_ERROR_FUNCTION(const char* file, const char* line) {
    GLenum errorCode = glGetError();
    if (errorCode != GL_NO_ERROR) {
        const GLubyte* errorString = gluErrorString(errorCode);
        exit_with_error("[file:%s line:%s] OpenGL Error: %s\n", 
                        file, line, (char*)errorString);
    }
}
