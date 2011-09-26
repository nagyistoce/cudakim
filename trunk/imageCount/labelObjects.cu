/*
 * labelObjects.cu
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
setImageByte( byte* dst, int stride, int x, int y, byte val)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row == x & col == y)
    dst[row * stride + col] = val;

}

__global__ void
dilateOriginalImageByte(byte* dst, byte* src, byte *org, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  byte pix01 = src[(row - 1) * width + col];
  byte pix10 = src[row * width + col - 1];
  byte pix11 = src[row * width + col];
  byte pix12 = src[row * width + col + 1];
  byte pix21 = src[(row + 1) * width + col];
  byte res;

  // Dilate morphological operation
  if ( (pix01 == 255) |
       (pix10 == 255) |
       (pix12 == 255) |
       (pix21 == 255) )
  {
	  res = 255;
  }
  else
  {
	  res = pix11;
  }

  // Anded original
  dst[row * width + col] = res & org[row * width + col];

}

__global__ void
diffImageReduced( byte* dst, byte* xk, byte *xk1, int stride)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  dst[row * stride + col] = abs(xk[row * stride + col] - xk1[row * stride + col]);

}

__global__ void
lableImageObject( byte* dst, byte* src, int stride, byte lable)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (src[row * stride + col] > 0)
	  dst[row * stride + col] = lable;
}

static bool findPoint(byte *img, ROI Size, int Stride, byte val, POINT *point)
{
	int x, y;
	bool found = false;

	// Search white pixel part of
	for (x = 0; x < Size.height; x++) {
		for (y = 0; y < Size.width; y++) {
			byte pixel = img[x*Stride + y];
			if (pixel >= val) {
				point->x = x;
				point->y = y;
				found = true;
				break;
			}
		}
	}
	return found;
}

static bool isImageBlank(byte *img, ROI Size, int Stride)
{
	int x, y;
	bool blank = true;

	// Search white pixel part of
	for (x = 0; x < Size.height; x++) {
		for (y = 0; y < Size.width; y++) {
			byte pixel = img[x*Stride + y];
			if (pixel > 0) {
				blank = false;
				break;
			}
		}
	}
	return blank;
}

float LabelObjects(byte *dst, byte *bw, ROI Size, int Stride)
{
	int n = 10;
	POINT point;
	int ImgResStride;
	size_t ImgDevStride;
	byte *ImgDevXk1;
	byte *ImgDevXk;
    byte *ImgDevA;
    byte *ImgDevTmp;
    byte *ImgDevXres;

    printf("[LabelObjects]\n");

    // Create result image on host
    byte *ImgTmp = MallocPlaneByte(Size.width, Size.height, &ImgResStride);
    byte *ImgXres = MallocPlaneByte(Size.width, Size.height, &ImgResStride);
    ImgResStride /= sizeof(byte);
    //printf("ImgResStride %d\n", ImgResStride);

    cutilSafeCall(cudaMemcpy2D(ImgXres, ImgResStride * sizeof(byte),
    						   bw, Stride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToHost) );

    // Create image A on device
    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevA), &ImgDevStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMemcpy2D(ImgDevA, ImgDevStride * sizeof(byte),
    						   bw, Stride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );


    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevXk1), &ImgDevStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevXk), &ImgDevStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevTmp), &ImgDevStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevXres), &ImgDevStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMemcpy2D(ImgDevXres, ImgDevStride * sizeof(byte),
    						   bw, Stride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );

    ImgDevStride /= sizeof(byte);
    //printf("ImgDevStride %d\n", ImgDevStride);

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    printf("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    printf("Threads in Block [%d,%d]\n", threads.x, threads.y);

    // start CUDA timer
    if (timerCUDA == 0) CreateTimer(&timerCUDA);
    RestartTimer(timerCUDA);

	// Select picture with one white pixel not yet found
    while (findPoint(ImgXres, Size, Stride, 255, &point))
    {

    	// Create picture with one white pixel
    	cutilSafeCall(cudaMemset2D(ImgDevXk1, ImgDevStride, 0, Size.width, Size.height)); //OK
    	setImageByte<<< grid, threads >>>(ImgDevXk1, ImgDevStride, point.x, point.y, 255); //OK

    	while (true)
    	{
    		// Dilate image anded with original
    		// Xk = imdilate(Xk_1, B) & A
    		dilateOriginalImageByte<<< grid, threads >>>(ImgDevXk, ImgDevXk1, ImgDevA, ImgDevStride);

    		// Compare if image are equal DiffBWImg(Xk, Xk_1) == 1 - reduced version kbe???
    		diffImageReduced<<< grid, threads >>>(ImgDevTmp, ImgDevXk, ImgDevXk1, ImgDevStride);
    		// Copy difference of images
    	    cutilSafeCall(cudaMemcpy2D(ImgTmp, ImgResStride * sizeof(byte),
    	    						   ImgDevTmp, ImgDevStride * sizeof(byte),
    	                               Size.width * sizeof(byte), Size.height,
    	                               cudaMemcpyDeviceToHost) );

    	    if (isImageBlank(ImgTmp, Size, Stride))
    	    {
    	    	//printf("Object %d found\n", n);
         	    // Images are equal
    	    	// h = find(Xk == 1); Xres(h) = n;
    	    	lableImageObject<<< grid, threads >>>(ImgDevXres, ImgDevXk, ImgDevStride, n);
    	    	n = n + 10;
    	    	break;
    	    }

  		    // Xk_1 = Xk
			cutilSafeCall(cudaMemcpy2D(ImgDevXk1, ImgDevStride * sizeof(byte),
									   ImgDevXk, ImgDevStride * sizeof(byte),
									   Size.width * sizeof(byte), Size.height,
									   cudaMemcpyDeviceToDevice) );
    	}

		// Copy result to host
	    cutilSafeCall(cudaMemcpy2D(ImgXres, ImgResStride * sizeof(byte),
	    						   ImgDevXres, ImgDevStride * sizeof(byte),
	                               Size.width * sizeof(byte), Size.height,
	                               cudaMemcpyDeviceToHost) );

    }

    StopTimer(timerCUDA);

    cutilCheckMsg("Kernel execution failed");

    cutilSafeCall(cudaMemcpy2D(dst, Stride * sizeof(byte),
    						   ImgXres, ImgResStride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToHost) );

    cutilSafeCall(cudaFree(ImgDevA));
    cutilSafeCall(cudaFree(ImgDevXk1));
    cutilSafeCall(cudaFree(ImgDevXk));
    cutilSafeCall(cudaFree(ImgDevTmp));
    cutilSafeCall(cudaFree(ImgDevXres));
	FreePlane(ImgTmp);
	FreePlane(ImgXres);

	// Find first
	return GetTimer(timerCUDA);
}

/* Matlab version of LabelObjects
 *
function [res] = DiffBWImg(X0, X1)
%% returns 0 if images are equal

diff = X0 - X1;
res = max(diff(:));

end

nhood = [0 1 0; 1 1 1; 0 1 0];
B = strel('arbitrary',nhood);

% Perform extraction of connected components
k = 1;
n = 2;
%A = bw1;
A = bw;
Xres = A;

while 1 % Finds all components

h = find(Xres == 1); % Vector for white pixels not yet found
if (isempty(h))
    break; % No more components found
end;

% Select picture with one white pixel not yet found
Xk_1 = zeros(size(bw));
Xk_1(h(1)) = 1;

while 1 % Finds one component
    Xk = imdilate(Xk_1, B) & A;
    if DiffBWImg(Xk, Xk_1) == 1 % not equal
        k = k + 1;
        Xk_1 = Xk;
    else % equal
        h = find(Xk == 1);
        Xres(h) = n;
        n = n + 1;
        break;
    end;
end;

end;

*/




