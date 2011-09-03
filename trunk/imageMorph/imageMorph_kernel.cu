/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* This sample is a templatized version of the template project.
 * It also shows how to correctly templatize dynamically allocated shared
 * memory arrays.
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include "sharedmem.cuh"


////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
template<class T>
__global__ void
testKernel( T* g_idata, T* g_odata) 
{
  // Shared mem size is determined by the host app at run time
  SharedMemory<T> smem;
  T* sdata = smem.getPointer();

  // access thread id
  const unsigned int tid = threadIdx.x;
  // access number of threads in this block
  const unsigned int num_threads = blockDim.x;

  // read in input data from global memory
  // use the bank checker macro to check for bank conflicts during host
  // emulation
  sdata[tid] = g_idata[tid];
  __syncthreads();

  // perform some computations
  sdata[tid] = (T) num_threads * sdata[tid];
  __syncthreads();

  // write data to global memory
  g_odata[tid] = sdata[tid];
}

__global__ void
erodeImage( float* dst, float* src, int width) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // read in input data from global memory
  // Structuring element
  float pix01 = src[(row - 1) * width + col];
  float pix10 = src[row * width + col - 1];
  float pix11 = src[row * width + col];
  float pix12 = src[row * width + col + 1];
  float pix21 = src[(row + 1) * width + col];

  // Erode morphological operation
  float sum = pix01 + pix10 + pix11 + pix12 + pix21;
  if (sum < 255*5) sum = 0;
  else sum = 255;
  
  dst[row * width + col] = sum;  
  
  __syncthreads();

}

__global__ void
tresholdImage( float* dst, float* src, int width) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (src[row * width + col] > 110) 
  	dst[row * width + col] = 255;
  else
  	dst[row * width + col] = 0;
  
  //__syncthreads();

}

__global__ void
diffImage( float *diff, float* dst, float* src, int width) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
    
  diff[row * width + col] = src[row * width + col] - dst[row * width + col];
  
  //__syncthreads();

}

__global__ void
copyImage( float* dst, float* src, int width) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
    
  dst[row * width + col] = src[row * width + col];
  
  //__syncthreads();

}

#endif // #ifndef _TEMPLATE_KERNEL_H_
