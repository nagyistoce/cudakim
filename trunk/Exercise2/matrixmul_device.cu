/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

// includes, system
#include <stdio.h>
#include <matrix.h>

// includes, project
#include <cutil_inline.h>


#define TILE_WIDTH 16

// global scope
// declare texture reference for 1D float texture
texture<float, 1> texM;
texture<float, 1> texN;

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P) {

	/// *** INSERT CODE ***
	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	float Pvalue = 0;
	for (int k = 0; k < M.width; ++k)
		Pvalue += M.elements[Row*M.width+k] * N.elements[k*N.width+Col];

	P.elements[Row*P.width+Col] = Pvalue;

}

// Matrix multiplication kernel texture thread specification
__global__ void MatrixMulTexKernel(int Mwidth, int Nwidth, Matrix P) {

	/// *** INSERT CODE ***
	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	float Pvalue = 0;
	for (int k = 0; k < Mwidth; ++k)
		Pvalue += tex1Dfetch(texM, Row*Mwidth+k) * tex1Dfetch(texN, k*Nwidth+Col);

	P.elements[Row*P.width+Col] = Pvalue;

}

////////////////////////////////////////////////////////////////////////////////
//! Optimized test kernel for device functionality using shared memory
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernelShared(Matrix M, Matrix N, Matrix P) {

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;

	float Pvalue = 0;

	for (int m = 0; m < (M.width + TILE_WIDTH - 1)/TILE_WIDTH; ++m) {

		Mds[ty][tx] = M.elements[Row*M.width + (m*TILE_WIDTH + tx)];
		Nds[ty][tx] = N.elements[(m*TILE_WIDTH + ty)*N.width + Col];
		__syncthreads();

		// Slow version - not symmetric
		//for (int k = 0; (k < TILE_WIDTH) && ((m*TILE_WIDTH) + k < M.width); ++k)

		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}

	P.elements[Row*P.width+Col] = Pvalue;
}

__global__ void MatrixMulKernelSharedOpt(Matrix M, Matrix N, Matrix P) {

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by*TILE_WIDTH + ty;
	int Col = bx*TILE_WIDTH + tx;

	float Pvalue = 0;

	//Load first file from registers to shared memory
	float Mdstmp = M.elements[Row*M.width + tx];
	float Ndstmp = N.elements[ty*N.width + Col];

	for (int m = 1; m < (M.width + TILE_WIDTH - 1)/TILE_WIDTH + 1 ; ++m) {

		// Deposit file from registers to shared memory
		Mds[ty][tx] = Mdstmp;
		Nds[ty][tx] = Ndstmp;
		__syncthreads();

		// Load next tile from global memory into registers
		Mdstmp = M.elements[Row*M.width + (m*TILE_WIDTH + tx)];
		Ndstmp = N.elements[(m*TILE_WIDTH + ty)*N.width + Col];

		// Compute current file
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__syncthreads();
	}

	P.elements[Row*P.width+Col] = Pvalue;
}

////////////////////////////////////////////////////////////////////////////////
// Timer functions
////////////////////////////////////////////////////////////////////////////////
void CreateTimer(unsigned int *timer)
{
    cutilCheckError(cutCreateTimer(timer));
    cutilCheckError(cutResetTimer(*timer));
}

void inline StartTimer(unsigned int timer)
{
    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStartTimer(timer));
}

void inline StopTimer(unsigned int timer)
{
    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStopTimer(timer));
}

void inline RestartTimer(unsigned int timer)
{
    cutilCheckError(cutResetTimer(timer));
    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStartTimer(timer));
}

float GetTimer(unsigned int timer)
{
	return cutGetAverageTimerValue(timer);
}

void DeleteTimer(unsigned int timer)
{
	cutilCheckError(cutDeleteTimer(timer));
}

////////////////////////////////////////////////////////////////////////////////
// Printing properties of NVIDIA GPU card
////////////////////////////////////////////////////////////////////////////////
void PrintDeviceProperties(void)
{
	int devID;
    cudaDeviceProp deviceProps;

	devID = cutGetMaxGflopsDeviceId();
    cudaSetDevice( devID );
	cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));

	printf("Device %d: [%s]\n", devID, deviceProps.name);
	printf("  Major revision number:            %d\n", deviceProps.major);
	printf("  Minor revision number:            %d\n", deviceProps.minor);
	printf("  Total amount of global memory:    %d\n", deviceProps.totalGlobalMem);
	printf("  Number of multiprocessors (SM):   %d\n", deviceProps.multiProcessorCount);
	printf("  Max. threads per multiprocessor   768\n"); // Gforce 9400 + G80
	printf("  Max. blocks per multiprocessor    8\n");   // Gforce 9400 + G80
	printf("  Execute multiple kernels:         %s\n", (deviceProps.concurrentKernels == 0 ? "no" : "yes"));
	printf("  Constant memory:                  %d\n", deviceProps.totalConstMem);
	printf("  Shared memory per block:          %d\n", deviceProps.sharedMemPerBlock);
	printf("  Registers per block:              %d\n", deviceProps.regsPerBlock);
	printf("  Warp size:                        %d\n", deviceProps.warpSize);
	printf("  Max. threads per block:           %d\n", deviceProps.maxThreadsPerBlock);
	printf("  Max. dimension of block:          [%d,%d,%d]\n", deviceProps.maxThreadsDim[0], deviceProps.maxThreadsDim[1], deviceProps.maxThreadsDim[2]);
	printf("  Max. dimension of grid:           [%d,%d,%d]\n", deviceProps.maxGridSize[0], deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);
	printf("  Max. memory pitch:                %d\n", deviceProps.memPitch);
	printf("  Texture alignment:                %d\n", deviceProps.textureAlignment);
	printf("  Clock rate:                       %d Hz\n", deviceProps.clockRate);
	printf("  Concurrent copy and exe:          %s\n", (deviceProps.deviceOverlap == 0 ? "no" : "yes"));
	printf("\n");

}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P) {
    // Load M and N to the device
    Matrix Md = AllocateDeviceMatrix(M);
    CopyToDeviceMatrix(Md, M);
    Matrix Nd = AllocateDeviceMatrix(N);
    CopyToDeviceMatrix(Nd, N);

    // Allocate P on the device
    Matrix Pd = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pd, P); // Clear memory
    Matrix Pod = AllocateDeviceMatrix(P);
    CopyToDeviceMatrix(Pod, P); // Clear memory

	// Setup the execution configuration
	/// *** INSERT CODE ***
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid( (P.width + dimBlock.x - 1) / dimBlock.x,
    		      (P.height + dimBlock.y -1) / dimBlock.y );

    printf("Matrix multiplication M[%d,%d]\n", M.width, M.height);
    printf("Matrix multiplication N[%d,%d]\n", N.width, N.height);
    printf("Matrix multiplication P[%d,%d]\n", P.width, P.height);
    printf("Grid (Blocks)    [%d,%d]\n", dimGrid.x, dimGrid.y);
    printf("Threads in Block [%d,%d]\n", dimBlock.x, dimBlock.y);

    //create and start CUDA timer
    unsigned int timerCUDA = 0;
    CreateTimer(&timerCUDA);

    // Launch the device computation threads!
    StartTimer(timerCUDA);
    MatrixMulKernel<<< dimGrid, dimBlock >>>(Md, Nd, Pd);
    //cutilCheckMsg("Kernel execution failed");
    StopTimer(timerCUDA);
    printf("Matrix multiplication processing time : %f ms \n", GetTimer(timerCUDA));

    // Create texture for N matrix
    const cudaChannelFormatDesc descN = cudaCreateChannelDesc<float>();
    size_t numN_bytes = Nd.width*Nd.height*sizeof(float);
    cudaBindTexture(NULL, &texN, (const void*)Nd.elements, &descN, numN_bytes);

    // Create texture for M matrix
    const cudaChannelFormatDesc descM = cudaCreateChannelDesc<float>();
    size_t numM_bytes = Md.width*Md.height*sizeof(float);
    cudaBindTexture(NULL, &texM, (const void*)Md.elements, &descM, numM_bytes);

    // Launch the device computation threads using shared memory
    RestartTimer(timerCUDA);
    MatrixMulTexKernel<<< dimGrid, dimBlock >>>(Md.width, Nd.width, Pod);
    StopTimer(timerCUDA);
    printf("Texture memory processing time : %f ms \n", GetTimer(timerCUDA));
    cudaUnbindTexture(texN);
    cudaUnbindTexture(texM);

    // Launch the device computation threads using shared memory
    RestartTimer(timerCUDA);
    MatrixMulKernelShared<<< dimGrid, dimBlock >>>(Md, Nd, Pd);
    //cutilCheckMsg("Kernel execution failed");
    StopTimer(timerCUDA);
    printf("Shared memory processing time : %f ms \n", GetTimer(timerCUDA));

    // Launch the device computation threads using shared memory with optimization
    RestartTimer(timerCUDA);
    MatrixMulKernelSharedOpt<<< dimGrid, dimBlock >>>(Md, Nd, Pd);
    StopTimer(timerCUDA);
    printf("Shared memory optimized processing time : %f ms \n", GetTimer(timerCUDA));

    DeleteTimer(timerCUDA);

    // Read P from the device
    CopyFromDeviceMatrix(P, Pod);

    // Free device matrices
    FreeDeviceMatrix(&Md);
    FreeDeviceMatrix(&Nd);
    FreeDeviceMatrix(&Pd);
    FreeDeviceMatrix(&Pod);
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
