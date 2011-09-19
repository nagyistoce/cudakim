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
#include "cutil_inline.h"


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
  if (sum < 255.0f*5) sum = 0;
  else sum = 255.0f;
  
  dst[row * width + col] = sum;  
  
}

__global__ void
dilateImage( float* dst, float* src, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float pix01 = src[(row - 1) * width + col];
  float pix10 = src[row * width + col - 1];
  float pix11 = src[row * width + col];
  float pix12 = src[row * width + col + 1];
  float pix21 = src[(row + 1) * width + col];

  // Dilate morphological operation
  if ( (pix01 >= 255.0f) |
       (pix10 >= 255.0f) |
       (pix12 >= 255.0f) |
       (pix21 >= 255.0f) )
  {
	  dst[row * width + col] = 255.0f;
  }
  else
  {
	  dst[row * width + col] = pix11;
  }

}

__global__ void
tresholdImage( float* dst, float* src, int width, int th)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (src[row * width + col] > th)
  	dst[row * width + col] = 255.0f;
  else
  	dst[row * width + col] = 0;
  
}

__global__ void
diffImage( float *diff, float* dst, float* src, int width) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
    
  diff[row * width + col] = src[row * width + col] - dst[row * width + col];
  
}

__global__ void
copyImage( float* dst, float* src, int width) 
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
    
  dst[row * width + col] = src[row * width + col];
  
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
