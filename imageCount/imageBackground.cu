/*
 * imageBackground.cu
 *
 * Finding background in a series of images in 3D, where the z-dimension is the time
 * the background image is found computing the median of the pixel intensity in the z-dimension
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, bmp utilities
#include "defs.h"
#include "BmpUtil.h"
#include "timer.h"

static unsigned int timerCUDA = 0;

__global__ void
test3DImages (byte* dst, int stride, cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
	  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	  byte* imgPtr = (byte *)devPitchedPtr.ptr;
	  size_t pitch = devPitchedPtr.pitch;
	  size_t slicePitch = pitch * height;
	  imgPtr += slicePitch*8;

	  byte cp = imgPtr[rowIdx * pitch + colIdx];

	  // Update average of images
	  dst[rowIdx * stride + colIdx] = cp;
}

// Sorts an array a of length depth
__device__ void insertionsort(byte *a, int depth)
{
	int i, j;
	byte t;
	for (i=1; i < depth; i++)
	{
		t = a[i];
		j = i-1;
		while(t < a[j] && j >= 0)
		{
			a[j+1] = a[j];
			j = j-1;
		}
		a[j+1] = t;
	}
}

__device__ void swap (byte *x, byte *y)
{
	byte tmp;
	tmp = *x;
	*x = *y;
	*y = tmp;
}

__device__ void bublesort (byte *a, int depth)
{
	int i, j;
	for (i = 0; i < (depth-1); i++)
		for (j = 0; j < (depth-(i+1)); j++)
			if (a[j] > a[j+1])
				swap(&a[j], &a[j+1]);
}


__global__ void
median3DImages (byte* dst, int stride, cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
	  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	  byte* imgPtr = (byte *)devPitchedPtr.ptr;
	  size_t pitch = devPitchedPtr.pitch;
	  size_t slicePitch = pitch * height;
	  byte median[DEPTH];

	  // Average of all images
	  for (int z = 0; z < depth; ++z)
	  {
		  byte *slice = imgPtr + z * slicePitch; // Find sliced image
		  byte *row = slice + rowIdx * pitch; // Find row in image
		  median[z] = row[colIdx];
	  }

	  //insertionsort(median, depth); // NOT WORKING ON MAC MINI
	  bublesort(median, depth);

	  // Update average of images
	  dst[rowIdx * stride + colIdx] = median[(DEPTH+1)/2];
}

// Find background image based on 3D cube of images
float ImageBackground(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride, int depth)
{
    byte *Dst;
    size_t DstStride;
    cudaMemcpy3DParms memcpy3DParms = {0};

    DEBUG_MSG("[ImageBackground]\n");

    // Create src pointer and extent
    memcpy3DParms.srcPtr = make_cudaPitchedPtr(ImgSrc, Stride, Size.width, Size.height);
    memcpy3DParms.extent = make_cudaExtent(Size.width * sizeof(byte), Size.height, depth);

    // Allocation of memory for 3D source images in byte format
    cutilSafeCall(cudaMalloc3D(&memcpy3DParms.dstPtr, memcpy3DParms.extent));

    DEBUG_MSG("srcPtr: pitch, xsize, ysize [%d,%d,%d]\n", memcpy3DParms.srcPtr.pitch, memcpy3DParms.srcPtr.xsize, memcpy3DParms.srcPtr.ysize);
    DEBUG_MSG("dstPtr: pitch, xsize, ysize [%d,%d,%d]\n", memcpy3DParms.dstPtr.pitch, memcpy3DParms.dstPtr.xsize, memcpy3DParms.dstPtr.ysize);

    // Copy images to device memory
    memcpy3DParms.kind = cudaMemcpyHostToDevice;
    cutilSafeCall(cudaMemcpy3D(&memcpy3DParms));

    // Allocation of memory for 2D destination image in byte format
    cutilSafeCall(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(byte), Size.height));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)Size.width / BLOCK_SIZE), ceil((float)Size.height / BLOCK_SIZE) );

    DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    if (timerCUDA == 0) CreateTimer(&timerCUDA);
    RestartTimer(timerCUDA);

    median3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);
    //test3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);

    StopTimer(timerCUDA);

    cutilSafeCall(cudaThreadSynchronize());

    cutilSafeCall(cudaMemcpy2D(ImgDst, Size.width * sizeof(byte),
                                Dst, DstStride * sizeof(byte),
                                Size.width * sizeof(byte), Size.height,
                                cudaMemcpyDeviceToHost) );

    cutilSafeCall(cudaFree(memcpy3DParms.dstPtr.ptr));

    return GetTimer(timerCUDA);;
}



