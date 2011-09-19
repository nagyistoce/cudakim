#ifndef _OPEN_GL_UTILS_
#define _OPEN_GL_UTILS_

#if defined __APPLE__
    #include <GLUT/glut.h>
    #include <OpenGL/gl.h>
#else
    #define GL_GLEXT_PROTOTYPES 1 
    #include <GL/gl.h>
    #include <GL/glut.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int isSupported(const char* ext);
void inforce_extention_support(const char* ext);

/**
 *  Should never be used in the code, use CHECK_FOR_GL_ERROR(); instead
 */
void CHECK_FOR_GL_ERROR_FUNCTION(const char* file, const char* line);

#ifdef __cplusplus
}
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/**
 *  Checks for Open GL errors and report if an error was detected.
 *  NOTE: This call must not be used between calls to glBegin and glEnd. 
 */
/* #if _DEBUG */
#define CHECK_FOR_GL_ERROR() CHECK_FOR_GL_ERROR_FUNCTION(__FILE__,TOSTRING(__LINE__))
/* #else */
/* #define CHECK_FOR_GL_ERROR(); */
/* #endif */

#endif /* _OPEN_GL_UTILS_ */
