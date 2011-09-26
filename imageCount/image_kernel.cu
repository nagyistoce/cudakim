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

// Number of images to analyze
#define DEPTH 9

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

	  insertionsort(median, depth);

	  // Update average of images
	  dst[rowIdx * stride + colIdx] = median[(DEPTH+1)/2];
}

__global__ void
test3DImages (byte* dst, int stride, cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
	  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	  byte* imgPtr = (byte *)devPitchedPtr.ptr;
	  size_t pitch = devPitchedPtr.pitch;
	  size_t slicePitch = pitch * height;
	  //float sum = 0;
	  imgPtr += slicePitch*8;

	  byte cp = imgPtr[rowIdx * pitch + colIdx];

	  // Update average of images
	  dst[rowIdx * stride + colIdx] = cp;
}

__global__ void
diffImageByte( byte* diff, byte* back, byte* src, int stride)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  diff[row * stride + col] = abs(src[row * stride + col] - back[row * stride + col]);

}

__global__ void
erodeImage( float* dst, float* src, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  // read in input data from global memory
  // Structuring element
  /*
  float pix01 = (row > 0 ? src[(row - 1) * width + col] : 0);
  float pix10 = (col > 0 ? src[row * width + col - 1] : 0);
  float pix11 = src[row * width + col];
  float pix12 = (col < width - 1 ? src[row * width + col + 1] : 0);
  float pix21 = (row < height - 1 ? src[(row + 1) * width + col] : 0);
  */
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
erodeImageByte( byte* dst, byte* src, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // read in input data from global memory
  // Structuring element
  byte pix01 = src[(row - 1) * width + col];
  byte pix10 = src[row * width + col - 1];
  byte pix11 = src[row * width + col];
  byte pix12 = src[row * width + col + 1];
  byte pix21 = src[(row + 1) * width + col];

  // Erode morphological operation
  float sum = pix01 + pix10 + pix11 + pix12 + pix21;
  byte pixel = 255;
  if (sum < 255.0f*5)
	  pixel = 0;

  dst[row * width + col] = pixel;

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
dilateImageByte( byte* dst, byte* src, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  byte pix01 = src[(row - 1) * width + col];
  byte pix10 = src[row * width + col - 1];
  byte pix11 = src[row * width + col];
  byte pix12 = src[row * width + col + 1];
  byte pix21 = src[(row + 1) * width + col];

  // Dilate morphological operation
  if ( (pix01 == 255) |
       (pix10 == 255) |
       (pix12 == 255) |
       (pix21 == 255) )
  {
	  dst[row * width + col] = 255;
  }
  else
  {
	  dst[row * width + col] = pix11;
  }

}

__global__ void
dilateSE5Image( float* dst, float* src, int width)
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
tresholdImageByte( byte* dst, byte* src, int width, byte th)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (src[row * width + col] > th)
  	dst[row * width + col] = 255;
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
