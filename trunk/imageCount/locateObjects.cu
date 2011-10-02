/*
 * locateObjects.cu
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
diffImageByte( byte* diff, byte* back, byte* src, int stride)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  diff[row * stride + col] = abs(src[row * stride + col] - back[row * stride + col]);

}

__global__ void
diffImageFloat( float *diff, float* dst, float* src, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  diff[row * width + col] = abs(src[row * width + col] - dst[row * width + col]);

}

__global__ void
erodeImageFloat( float* dst, float* src, int width)
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
dilateImageFloat( float* dst, float* src, int width)
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
dilate5SEImageByte( byte* dst, byte* src, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  byte pix11 = src[row * width + col];

  // Dilate morphological operation SE[5x5]
  if (
	   (src[(row - 2) * width + col - 1] == 255) |
	   (src[(row - 2) * width + col] == 255) |
	   (src[(row - 2) * width + col + 1] == 255) |
	   (src[(row - 1) * width + col - 1] == 255) |
	   (src[(row - 1) * width + col] == 255) |
	   (src[(row - 1) * width + col + 1] == 255) |
	   (src[row * width + col - 2] == 255) |
	   (src[row * width + col - 1] == 255) |
       (src[row * width + col + 1] == 255) |
       (src[row * width + col + 2] == 255) |
       (src[(row + 1) * width + col - 1] == 255) |
       (src[(row + 1) * width + col] == 255) |
       (src[(row + 1) * width + col + 1] == 255) |
       (src[(row + 2) * width + col - 1] == 255) |
       (src[(row + 2) * width + col] == 255) |
       (src[(row + 2) * width + col + 1] == 255)
       )
  {
	  dst[row * width + col] = 255;
  }
  else
  {
	  dst[row * width + col] = pix11;
  }

}

__global__ void
tresholdImageFloat( float* dst, float* src, int width, int th)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (src[row * width + col] > th)
  	dst[row * width + col] = 255.0f;
  else
  	dst[row * width + col] = 0;

}

__global__ void
tresholdImageByte( byte* dst, byte* src, int strideDst, int strideSrc, byte th)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (src[row * strideSrc + col] > th)
  	dst[row * strideDst + col] = 255;
  else
  	dst[row * strideDst + col] = 0;

}

__global__ void
copyImageFloat( float* dst, float* src, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  dst[row * width + col] = src[row * width + col];

}

// Compute difference between 2 images
float DiffImages(byte *ImgDst, byte *ImgBack, byte *ImgSrc, ROI Size, int ISStride, int IBStride)
{
    byte  *Diff, *Back, *Src;
    size_t DiffStride, SrcStride, BackStride;

    DEBUG_MSG("[DiffImages]\n");

    // Allocation of device memory for 2D difference image
    cutilSafeCall(cudaMallocPitch((void **)(&Diff), &DiffStride, Size.width * sizeof(byte), Size.height));
    DiffStride /= sizeof(byte);
    //DEBUG_MSG("DiffStride %d\n", DiffStride);

    // Allocation of memory for 2D background and source image in byte format
    cutilSafeCall(cudaMallocPitch((void **)(&Back), &BackStride, Size.width * sizeof(byte), Size.height));
    BackStride /= sizeof(byte);
    //DEBUG_MSG("BackStride %d\n", BackStride);

    cutilSafeCall(cudaMallocPitch((void **)(&Src), &SrcStride, Size.width * sizeof(byte), Size.height));
    SrcStride /= sizeof(byte);
    //DEBUG_MSG("SrcStride %d\n", SrcStride);

    //copy background image from host memory to device
    cutilSafeCall(cudaMemcpy2D(Back, BackStride * sizeof(byte),
                               ImgBack, IBStride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );

    //copy source image from host memory to device
    cutilSafeCall(cudaMemcpy2D(Src, SrcStride * sizeof(byte),
                               ImgSrc, ISStride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)Size.width / BLOCK_SIZE), ceil((float)Size.height / BLOCK_SIZE) );

    DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    if (timerCUDA == 0) CreateTimer(&timerCUDA);
    RestartTimer(timerCUDA);
    diffImageByte<<< grid, threads >>>(Diff, Back, Src, SrcStride);
    StopTimer(timerCUDA);

    cutilSafeCall(cudaMemcpy2D(ImgDst, IBStride * sizeof(byte),
                                Diff, DiffStride * sizeof(byte),
                                Size.width * sizeof(byte), Size.height,
                                cudaMemcpyDeviceToHost) );

    //clean up memory
    cutilSafeCall(cudaFree(Diff));
    cutilSafeCall(cudaFree(Back));
    cutilSafeCall(cudaFree(Src));

    return GetTimer(timerCUDA);
}

#define BOARDER_SIZE		4 // Additional boarder added to image for dilation, errosion and blurring
#define ADD_BOARDER(ptr, stride) (ptr + stride*BOARDER_SIZE*sizeof(byte) + BOARDER_SIZE*sizeof(byte));

// Performs thresholding and morphological operations like dilation and erode of image
float MorphObjects(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride)
{
    byte *Src, *DstBW, *DstBWb, *Dst1, *Dst1b, *Dst2, *Dst2b;
    size_t DstStride, SrcStride;

    DEBUG_MSG("[MorphObjects]\n");

    // Allocation of memory for 2D source image in single precision format
    cutilSafeCall(cudaMallocPitch((void **)(&Src), &SrcStride, Size.width * sizeof(byte), Size.height));
    SrcStride /= sizeof(byte);
    //DEBUG_MSG("SrcStride %d\n", SrcStride);

    //copy source image from host memory to device
    cutilSafeCall(cudaMemcpy2D(Src, SrcStride * sizeof(byte),
                               ImgSrc, Stride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );

    // Allocation of device memory for 2D destination image in single precision format
    ROI SB = Size;
    SB.width += BOARDER_SIZE*2; // Add black boarders to allocated device image memory buffers
    SB.height += BOARDER_SIZE*2;

    cutilSafeCall(cudaMallocPitch((void **)(&DstBW), &DstStride, SB.width * sizeof(byte), SB.height));
    cutilSafeCall(cudaMallocPitch((void **)(&Dst1), &DstStride, SB.width * sizeof(byte), SB.height));
    cutilSafeCall(cudaMallocPitch((void **)(&Dst2), &DstStride, SB.width * sizeof(byte), SB.height));

    //DEBUG_MSG("DstStride %d\n", DstStride);
    // Clear device memory for all images
    cutilSafeCall(cudaMemset2D((void *)(DstBW), DstStride, 0, SB.width * sizeof(byte), SB.height));
    cutilSafeCall(cudaMemset2D((void *)(Dst1), DstStride, 0, SB.width * sizeof(byte), SB.height));
    cutilSafeCall(cudaMemset2D((void *)(Dst2), DstStride, 0, SB.width * sizeof(byte), SB.height));
    DstStride /= sizeof(byte);

    // Add black boarder to all images in 2D device memory
    // Needed since erode and dilate image structuring elements SE are [3x3] and [5x5]
    DstBWb = ADD_BOARDER(DstBW, DstStride);
    Dst1b = ADD_BOARDER(Dst1, DstStride);
    Dst2b = ADD_BOARDER(Dst2, DstStride);

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)Size.width / BLOCK_SIZE), ceil((float)Size.height / BLOCK_SIZE) );

    DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    // start CUDA timer
    if (timerCUDA == 0) CreateTimer(&timerCUDA);
    RestartTimer(timerCUDA);

    // Generate BW image
    tresholdImageByte<<< grid, threads >>>(DstBWb, Src, DstStride, SrcStride, 25);
    cutilSafeCall(cudaThreadSynchronize());

    // Erode image with structuring element
    erodeImageByte<<< grid, threads >>>(Dst1b, DstBWb, DstStride);
    // Dilate image with structuring element
    //dilateImageByte<<< grid, threads >>>(Dst2, Dst1, DstStride);
    dilate5SEImageByte<<< grid, threads >>>(Dst2b, Dst1b, DstStride);

    StopTimer(timerCUDA);

    cutilCheckMsg("Kernel execution failed");

    //copy eroded image from device memory to host memory in Src
    cutilSafeCall(cudaMemcpy2D(ImgDst, Stride * sizeof(byte),
                                Dst2b, DstStride * sizeof(byte),
                                Size.width * sizeof(byte), Size.height,
                                cudaMemcpyDeviceToHost) );

    //clean up memory
    cutilSafeCall(cudaFree(Src));
    cutilSafeCall(cudaFree(DstBW));
    cutilSafeCall(cudaFree(Dst1));
    cutilSafeCall(cudaFree(Dst2));

    //return time taken by the operation
    return GetTimer(timerCUDA);
}

// Performs thresholding and morphological operations like dilation and erode of image
// Converts to floating point format before
float MorphObjectsFloat(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride)
{
    float *Dst, *DstBW, *Src, *Diff;
    size_t DstStride, SrcStride, DiffStride;

    DEBUG_MSG("[MorphObjectsFloat]\n");

    //convert source image to float representation
    int ImgSrcFStride;
    float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
    CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);

    // Allocation of memory for 2D source image in single precision format
    cutilSafeCall(cudaMallocPitch((void **)(&Src), &SrcStride, Size.width * sizeof(float), Size.height));
    SrcStride /= sizeof(float);
    //DEBUG_MSG("SrcStride %d\n", SrcStride);

    //copy source image from host memory to device
    cutilSafeCall(cudaMemcpy2D(Src, SrcStride * sizeof(float),
                               ImgSrcF, ImgSrcFStride * sizeof(float),
                               Size.width * sizeof(float), Size.height,
                               cudaMemcpyHostToDevice) );

    // Allocation of device memory for 2D destination image in single precision format
    cutilSafeCall(cudaMallocPitch((void **)(&DstBW), &DstStride, Size.width * sizeof(float), Size.height));
    cutilSafeCall(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));
    DstStride /= sizeof(float);

    cutilSafeCall(cudaMallocPitch((void **)(&Diff), &DiffStride, Size.width * sizeof(float), Size.height));
    DiffStride /= sizeof(float);

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    //create and start CUDA timer
    if (timerCUDA == 0) CreateTimer(&timerCUDA);
    RestartTimer(timerCUDA);

    //copy image from device memory to device memory
    /*
    cutilSafeCall(cudaMemcpy2D(Dst, DstStride * sizeof(float),
                                Src, SrcStride * sizeof(float),
                                Size.width * sizeof(float), Size.height,
                                cudaMemcpyDeviceToDevice) );

    copyImage<<< grid, threads >>>(Dst, Src, Size.width);
    */

    // Generate BW image
    //tresholdImage<<< grid, threads >>>(DstBW, Src, Size.width, 110);
    tresholdImageFloat<<< grid, threads >>>(DstBW, Src, DstStride, 15);
    cutilSafeCall(cudaThreadSynchronize());

    // Erode image with structuring element
    erodeImageFloat<<< grid, threads >>>(Dst, DstBW, DstStride);
    // Dilate image with structuring element
    dilateImageFloat<<< grid, threads >>>(Diff, Dst, DstStride);
    //dilateImage<<< grid, threads >>>(Diff, DstBW, DstStride);
    cutilSafeCall(cudaThreadSynchronize());

    // Diff BW and eroded image
    //diffImageFloat<<< grid, threads >>>(Diff, DstBW, Dst, Size.width);
    //cutilSafeCall(cudaThreadSynchronize());

    StopTimer(timerCUDA);

    cutilCheckMsg("Kernel execution failed");

    //copy eroded image from device memory to host memory in Src
    cutilSafeCall(cudaMemcpy2D(ImgSrcF, ImgSrcFStride * sizeof(float),
                                Diff, DiffStride * sizeof(float),
                                Size.width * sizeof(float), Size.height,
                                cudaMemcpyDeviceToHost) );

    CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

    //clean up memory
    cutilSafeCall(cudaFree(Src));
    cutilSafeCall(cudaFree(Dst));
    cutilSafeCall(cudaFree(DstBW));
    cutilSafeCall(cudaFree(Diff));
    FreePlane(ImgSrcF);

    //return time taken by the operation
    return GetTimer(timerCUDA);
}

