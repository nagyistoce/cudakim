#include "rd_kernel.h"

#include <stdio.h>
#include <timer.h>

// CUDA timer definition
unsigned int timerCUDA = 0;

// global scope
// declare texture reference for 1D float texture
texture<float, 1> texU;
texture<float, 1> texV;

/*
 * Utility function to initialize U and V
*/
__host__
void initializeConcentrations(unsigned int width, unsigned int height, float *U, float *V) {
		float *_U = new float[width*height];
		float *_V = new float[width*height];

		int k = 0;
		int i, j;

		for (i = 0; i < width * height; ++i) {
			_U[k] = 1.0f;
			_V[k++] = 0.0f;
		}

		for (i = (0.48f)*height; i < (0.52f)*height; ++i) {
			for (j = (0.48f)*width; j < (0.52f)*width; ++j) {
				_U[ (i * width + j) ] = 0.5f;
				_V[ (i * width + j) ] = 0.25f;
			}
		}

		// Now perturb the entire grid. Bound the values by [0,1]
		for (k = 0; k < width * height; ++k) {
			if ( _U[k] < 1.0f ) {
				float rRand = 0.02f*(float)rand() / RAND_MAX - 0.01f;
				_U[k] += rRand * _U[k];
			}
			if ( _V[k] < 1.0f ) {
				float rRand = 0.02f*(float)rand() / RAND_MAX - 0.01f;
				_V[k] += rRand * _V[k];
			}
		}

		// Upload initial state U and V to the GPU
		cudaMemcpy( U, _U, width*height*sizeof(float), cudaMemcpyHostToDevice );
		cudaMemcpy( V, _V, width*height*sizeof(float), cudaMemcpyHostToDevice );

		delete[] _U;
		delete[] _V;
}

/*
 * Kernel for the reaction-diffusion model
 * This kernel is responsible for updating 'U' and 'V'
 */
__global__
void rd_kernel(unsigned int width, unsigned int height,
               float dt, float dx, float Du, float Dv,
               float F, float k, float *U, float *V) {

	// Coordinate of the current pixel (for this thread)
	const uint2 co = make_uint2( blockIdx.x*blockDim.x + threadIdx.x,
                                 blockIdx.y*blockDim.y + threadIdx.y );
	
	// Linear index of the curernt pixel
	const unsigned int idx = co.y*width + co.x;

	//
	// REACTION-DIFFUSION KERNEL - Kim Bjerge's version
	//
	// done - Notes: - optimization - kernel without "if"
	// done - Texture version for Mac
	// done - Meassurments - time
	// done - Ressourcer forbrug?

	// Tile assymetrisk ?
	// Kernel with shared memory how ?

        // Use registeres to save current values of U and V
        float Ui = U[idx];
        float Vi = V[idx];

        // Skip computing first and last line in image
        if (idx >= width && idx < width*(height-1))
        {
		// Computes the Laplacian operator for U and V - used values in x and y dimensions
		//float laplacianU = Ui;
		//float laplacianV = Vi;
		float laplacianU = (U[idx+1] + U[idx-1] + U[idx+width] + U[idx-width] - 4 * Ui)/(dx*dx);
		float laplacianV = (V[idx+1] + V[idx-1] + V[idx+width] + V[idx-width] - 4 * Vi)/(dx*dx);


		// Computes the diffusion and reaction of the two chemicals reactants mixed together
		float Uf = Du * laplacianU - Ui*powf(Vi,2) + F*(1 - Ui);
		//float Uf = Du * laplacianU; // Difusion only
		float Vf = Dv * laplacianV + Ui*powf(Vi,2) - (F + k)*Vi;

		U[idx] = Ui + dt*Uf;
		V[idx] = Vi + dt*Vf;
        }
        
}

/*
 * Optimized kernel for the reaction-diffusion model
 * Using texture memory for U and V
 * This kernel is responsible for updating 'U' and 'V'
 */
__global__
void rd_kernel_tex(unsigned int width, unsigned int height,
               float dt, float dx, float Du, float Dv,
               float F, float k, float *U, float *V) {

	// Coordinate of the current pixel (for this thread)
	const uint2 co = make_uint2( blockIdx.x*blockDim.x + threadIdx.x,
                                 blockIdx.y*blockDim.y + threadIdx.y );

	// Linear index of the curernt pixel
	const unsigned int idx = co.y*width + co.x;

	//
	// REACTION-DIFFUSION KERNEL - Kim Bjerge's version
	//

	// Use registeres to save current values of U and V

    float Ui = tex1Dfetch(texU, idx);
	float Vi = tex1Dfetch(texV, idx);

	// Skip computing first and last line in image
	if (idx >= width && idx < width*(height-1))
	{
		// Computes the Laplacian operator for U and V - used values in x and y dimensions
		float laplacianU = (tex1Dfetch(texU, idx+1) + tex1Dfetch(texU,idx-1) + tex1Dfetch(texU, idx+width) + tex1Dfetch(texU, idx-width) - 4 * Ui)/(dx*dx);
		float laplacianV = (tex1Dfetch(texV, idx+1) + tex1Dfetch(texV, idx-1) + tex1Dfetch(texV, idx+width) + tex1Dfetch(texV, idx-width) - 4 * Vi)/(dx*dx);


		// Computes the diffusion and reaction of the two chemicals reactants mixed together
		float Uf = Du * laplacianU - Ui*powf(Vi,2) + F*(1 - Ui);
		float Vf = Dv * laplacianV + Ui*powf(Vi,2) - (F + k)*Vi;

		U[idx] = Ui + dt*Uf;
		V[idx] = Vi + dt*Vf;
	}

}

/*
 * Kernel for the reaction-diffusion model
 * This kernel is responsible for updating 'U' and 'V'
 */
__global__
void rd_kernel_opt1(unsigned int width, unsigned int height,
               float dt, float dx, float Du, float Dv,
               float F, float k, float *U, float *V) {

	// Coordinate of the current pixel (for this thread)
	const uint2 co = make_uint2( blockIdx.x*blockDim.x + threadIdx.x,
                                 blockIdx.y*blockDim.y + threadIdx.y );

	// Linear index of the curernt pixel
	const unsigned int idx = co.y*width + co.x;

	//
	// REACTION-DIFFUSION KERNEL - Optimized version 1
	//

	// Use registeres to save current values of U and V
	float Ui = U[idx];
	float Vi = V[idx];

	// Computes the Laplacian operator for U and V - used values in x and y dimensions
	float laplacianU = (U[idx+1] + U[idx-1] + U[idx+width] + U[idx-width] - 4 * Ui)/(dx*dx);
	float laplacianV = (V[idx+1] + V[idx-1] + V[idx+width] + V[idx-width] - 4 * Vi)/(dx*dx);


	// Computes the diffusion and reaction of the two chemicals reactants mixed together
	float Uf = Du * laplacianU - Ui*powf(Vi,2) + F*(1 - Ui);
	float Vf = Dv * laplacianV + Ui*powf(Vi,2) - (F + k)*Vi;

	// Needed since U and V values used by all threads in block
	__syncthreads();

	U[idx] = Ui + dt*Uf;
	V[idx] = Vi + dt*Vf;
}

/*
 * Kernel for the reaction-diffusion model
 * This kernel is responsible for updating 'U' and 'V'
 */
__global__
void rd_kernel_opt2(unsigned int width, unsigned int height,
               float dt, float dx, float Du, float Dv,
               float F, float k, float *U, float *V) {

	// Coordinate of the current pixel (for this thread)
	const uint2 co = make_uint2( blockIdx.x*blockDim.x + threadIdx.x,
                                 blockIdx.y*blockDim.y + threadIdx.y );

	// Linear index of the curernt pixel
	const unsigned int idx = co.y*width + co.x;

	// REACTION-DIFFUSION KERNEL - Optimized version 2
	// Use registeres to save current values of U and V

	U[idx] = U[idx] + dt*(Du * ((U[idx+1] + U[idx-1] + U[idx+width] + U[idx-width] - 4 * U[idx])/(dx*dx)) - U[idx]*V[idx]*V[idx] + F*(1 - U[idx]));
	V[idx] = V[idx] + dt*(Dv * ((V[idx+1] + V[idx-1] + V[idx+width] + V[idx-width] - 4 * V[idx])/(dx*dx)) + U[idx]*V[idx]*V[idx] - (F + k)*V[idx]);
}

/*
 * Wrapper for the reaction-diffusion kernel. 
 * Called every frame by 'display'
 * 'result_devPtr' is a floating buffer used for visualization.
 * Make sure whatever needs visualization goes there.
 */
extern "C" __host__
void rd(unsigned int width, unsigned int height, float *result_devPtr) {
	// Create buffers for 'U' and 'V' at first pass
	static float *U, *V;
	static bool first_pass = true;

	if (first_pass){
		// Allocate device memory for U and V
		cudaMalloc((void**)&U, width*height*sizeof(float));
		cudaMalloc((void**)&V, width*height*sizeof(float));
 
		// Check for Cuda errors
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("\nCuda error detected: %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
			exit(1);
		}

		// Initialize U and V on the CPU and upload to the GPU
		initializeConcentrations( width, height, U, V );

		CreateTimer(&timerCUDA);

		// Make sure we never get in here again...
		first_pass = false;
	}

	// Kernel block dimensions
	const dim3 blockDim(16,16);

	// Verify input image dimensions
	if (width%blockDim.x || height%blockDim.y) {
		printf("\nImage width and height must be a multiple of the block dimensions\n");
		exit(1);
	}

	// Experiment with different settings of these constants
        /* Original values
	const float dt = 1.0f;
	const float dx = 2.0f;
	const float Du = 0.0004f*((width*height)/100.0f);
	const float Dv = 0.0002f*((width*height)/100.0f);
	const float F = 0.012f; 
	const float k = 0.052f;
        */

	const float dt = 0.2f;
	const float dx = 2.0f;
	const float Du = 0.0004f*((width*height)/100.0f);
	const float Dv = 0.0002f*((width*height)/100.0f); // Impact on how fast V diffusses (0.0001 or 0.0002)
	const float F = 0.012f; 
	const float k = 0.052f;


	// Invoke kernel (update U and V)
#if 1 // Optimized skipping top and bottom edges
	RestartTimer(timerCUDA);
	//rd_kernel<<< dim3(width/blockDim.x, height/blockDim.y), blockDim >>>( width, height, dt, dx, Du, Dv, F, k, U, V );
	//rd_kernel_opt1<<< dim3(width/blockDim.x, (height-2)/blockDim.y), blockDim >>>( width, height-2, dt, dx, Du, Dv, F, k, &U[width], &V[width] );
	rd_kernel_opt2<<< dim3(width/blockDim.x, (height-2)/blockDim.y), blockDim >>>( width, height-2, dt, dx, Du, Dv, F, k, &U[width], &V[width] );
	StopTimer(timerCUDA);
	float average = GetAverage(timerCUDA);
	if (average > 0)
	   printf("Opt2 %f ms\n", average);
#endif

#if 0 // Optimized with texture memory
    // Create texture for U matrix
    const cudaChannelFormatDesc descU = cudaCreateChannelDesc<float>();
    size_t numU_bytes = width*height*sizeof(float);
    cudaBindTexture(NULL, &texU, (const void*)U, &descU, numU_bytes);

    // Create texture for V matrix
    const cudaChannelFormatDesc descV = cudaCreateChannelDesc<float>();
    size_t numV_bytes = width*height*sizeof(float);
    cudaBindTexture(NULL, &texV, (const void*)V, &descV, numV_bytes);

    RestartTimer(timerCUDA);
	rd_kernel_tex<<< dim3(width/blockDim.x, height/blockDim.y), blockDim >>>( width, height, dt, dx, Du, Dv, F, k, U, V);
	StopTimer(timerCUDA);
	float average = GetAverage(timerCUDA);
	if (average > 0)
		printf("Tex %f ms\n", average);

    cudaUnbindTexture(texU);
    cudaUnbindTexture(texV);
#endif

	// Check for errors
	cudaError_t err = cudaGetLastError();
	if( err != cudaSuccess ){
		printf("\nCuda error detected in 'rd_kernel': %s. Quitting.\n", cudaGetErrorString(err) ); fflush(stdout);
		exit(1);
	}

	// For visualization we use a 'float1' image. You can use either 'U' or 'V'.
	cudaMemcpy( result_devPtr, V, width*height*sizeof(float), cudaMemcpyDeviceToDevice );
}
