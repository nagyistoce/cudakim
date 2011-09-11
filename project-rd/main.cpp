#if defined(__APPLE__) || defined(MACOSX)
// Mac includes

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GLUT/glut.h>

#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>

#include <rendercheck_gl.h>

#else

// Linux includes 
#include <GL/glew.h>
#include <opengl/opengl_utilities.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cutil.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#endif

#include <rd_kernel.h>

const unsigned int width = 512, height = 512;
//const unsigned int width = 1024, height = 1024;
unsigned int pbo_x=0, pbo_y=0;

GLuint pbo;    /* OpenGL pixel buffer object (map between Cuda and OpenGL) */
GLuint texid;  /* Texture (display pbo) */
GLuint shader; /* Pixel shader for rendering of texture */

/*
 * Glut display callback function to render images using OpenGL
 * The display function is automatically called by Glut in each frame
 */
void display() {
	/* Map the OpenGL pixel buffer object (PBO) to Cuda device memory */
	float *devPtr;
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&devPtr, pbo));

	if( !devPtr ){
		printf("\nWARNING: no pbo allocated for reconstruction result!\n");
	}
	if( pbo_x==0 || pbo_y == 0 ){
		printf("\nWARNING: pbo buffer size is 0! Has initializeOpenGL() been called?\n");
	}

	/*
		Compute reaction-diffusion model. 
		----------------------------------
		Save result in device pointer 'devPtr'
	*/
	rd( pbo_x, pbo_y, devPtr );

	/* Unmap Cuda <-> PBO relation */
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

	/* Display results using standard OpenGL texture mapping of a quad. */

    /* Clear window */
	glClear(GL_COLOR_BUFFER_BIT);

	/* Copy data from PBO to texture */
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, texid);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pbo_x, pbo_y, GL_LUMINANCE, GL_FLOAT, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	/* Use simple fragment program to display the floating point texture */
	glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glDisable(GL_DEPTH_TEST);

	/* Render image */
	glBegin(GL_QUADS);
    glVertex2f(0, 0); glTexCoord2f(0, 0);
    glVertex2f(0, 1); glTexCoord2f(0, 1);
    glVertex2f(1, 1); glTexCoord2f(1, 1);
    glVertex2f(1, 0); glTexCoord2f(1, 0);
	glEnd();

	/* Restore original state */
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_FRAGMENT_PROGRAM_ARB);

	/* Swap the doubleBuffer to see the result */
	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
	switch(key) {
    case 27: /* ESCAPE */
        exit(0);
        break;
    default:
        break;
	}
}

void idle() {
    glutPostRedisplay();
}

void reshape(int x, int y) {
    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void cleanup() {
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
	glDeleteBuffersARB(1, &pbo);
    glDeleteTextures(1, &texid);
    glDeleteProgramsARB(1, &shader);
}

void initializeGlut(int argc, char** argv, 
                    unsigned int width, unsigned int height) {
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Reaction diffusion");
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);
}

void initializeGlew() {
	glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(1);
    }
}

bool initializeCuda() {
	printf("\n"); fflush(stdout);

	cudaDeviceProp deviceProp;  
	unsigned int device = 0;

	CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, device));
    if (deviceProp.major < 1) {
        fprintf(stderr, "cutil error: device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Using device %d: %s\n", device, deviceProp.name);
    CUDA_SAFE_CALL(cudaSetDevice(device));

    return true;
}

GLuint compileASMShader(GLenum program_type, const char *code) {
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
    if (error_pos != -1) {
        const GLubyte *error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }
    return program_id;
}

void initializeOpenGL(unsigned int width, unsigned int height) {
	while (glGetError() != GL_NO_ERROR) {
		printf("\nWARNING: glError detected prior to initializeOpenGL");
		fflush(stdout);
	}

	/* Create pixel buffer object (PBO) to "render Cuda memory" through a texture */
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

	/* Initialize PBO with zero image */
	float *tmp = (float*) calloc( width*height, sizeof(float) );
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(float), tmp, GL_STREAM_DRAW_ARB);

	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));

	pbo_x = width;
	pbo_y = height;

    /* Create texture for display */
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D, texid);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, width, height, 0, GL_LUMINANCE, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    
	glBindTexture(GL_TEXTURE_2D, 0);

    /* Shader for displaying floating-point texture */
    static const char *shader_code = 
        "!!ARBfp1.0\n"
        "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
        "END";

    /* Load shader program */
    shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

	while (glGetError() != GL_NO_ERROR) {
		printf("\nWARNING: glError detected at the end of initializeOpenGL");
		fflush(stdout);
	}

	free(tmp);
}

int main( int argc, char** argv) {
	/* Initialize GLUT */
	initializeGlut( argc, argv, width, height );

	/* Initialize GLEW */
	initializeGlew();

    /* Initialize Cuda and check for support */
	if (!initializeCuda()) {
		printf("\nERROR: Cannot initialize Cuda!\n");
		exit(1);
	}

    /* Initialize OpenGL, PBO, and the texture */
	initializeOpenGL( width, height );

	/* Call cleanup at exit */
	atexit(cleanup);

    glutMainLoop();
    return 0;
}
