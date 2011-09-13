/*
	SimpleGL SDK example modified to prepare for a simple 
	Spring-mass-damper model implementation.
*/

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* 
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes
#include <cutil.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1024;
const unsigned int window_height = 1024;

const unsigned int mesh_width = 30;
const unsigned int mesh_height = 30;
const unsigned int mesh_depth = 30;

// vbo variables (two input, one output)
GLuint old_vbo, cur_vbo, new_vbo;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
float user_a = -0.0018f;
float user_dt = 0.5f;

////////////////////////////////////////////////////////////////////////////////
// kernels
#include <simpleGL_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createVBO( GLuint* vbo);
void deleteVBO( GLuint* vbo);
void initializeVBO(GLuint vbo, float3 offset);

// rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda();

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    runTest( argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "Cuda MSD");

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);

    // create VBOs
    createVBO( &old_vbo);
    createVBO( &cur_vbo);
    createVBO( &new_vbo);

	// initial position of box
	const float3 offset = make_float3( 0.0f, 2.5f, 0.0f );
	initializeVBO(old_vbo, offset);
	initializeVBO(cur_vbo, offset);

	// run the cuda part
    runCuda();

    // start rendering mainloop
    glutMainLoop();
}

void initializeVBO(GLuint vbo, float3 offset)
{
    // map OpenGL buffer objects for writing from CUDA
    float4 *dataPtr;
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dataPtr, vbo));

    // execute the kernel
	const unsigned int block_dim_x = 256;
	dim3 block(block_dim_x, 1, 1);
    dim3 grid(ceil((float)(mesh_width*mesh_height*mesh_depth)/(float)block_dim_x), 1, 1);
    msd_initialize_kernel<<< grid, block>>>(dataPtr, offset, make_uint3(mesh_width, mesh_height, mesh_depth));

    // unmap buffer object
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo));
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
    // map OpenGL buffer objects for writing from CUDA
    float4 *_old_pos, *_cur_pos, *_new_pos;
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&_old_pos, old_vbo));
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&_cur_pos, cur_vbo));
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&_new_pos, new_vbo));

    // execute the kernel
	const unsigned int block_dim_x = 256;
	dim3 block(block_dim_x, 1, 1);
    dim3 grid(ceil((float)(mesh_width*mesh_height*mesh_depth)/(float)block_dim_x), 1, 1);
    msd_kernel<<< grid, block>>>(_old_pos, _cur_pos, _new_pos, make_uint3(mesh_width, mesh_height, mesh_depth), user_a, user_dt);

    // unmap buffer object
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( new_vbo));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( cur_vbo));
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject( old_vbo));

	// cycle buffers
	GLuint tmp = old_vbo;
	old_vbo = cur_vbo;
	cur_vbo = new_vbo;
	new_vbo = tmp;
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint* vbo)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * mesh_depth * 4 * sizeof( float);
    glBufferData( GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vbo));

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO( GLuint* vbo)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vbo));

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions
    runCuda();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, -1.0f, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, old_vbo); // old_vbo is the one that was written to last
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height * mesh_depth );
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {

		case( 27) :
			deleteVBO( &old_vbo);
			deleteVBO( &cur_vbo);
			deleteVBO( &new_vbo);
			exit( 0);
		case 'A':
			user_a -= 0.0001f;
			printf("Gravity %f\n", user_a);
			break;
		case 'a':
			user_a += 0.0001f;
			printf("Gravity %f\n", user_a);
			break;
		case 'T':
			user_dt += 0.01f;
			printf("Time %f\n", user_dt);
			break;
		case 't':
			user_dt -= 0.01f;
			printf("Time %f\n", user_dt);
			break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}
