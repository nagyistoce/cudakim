/*
 * labelObjectsOpt.cu
 *
 * Finds connected components in binary (BW) image
 * using 8 or 4 neighbor pixels
 *
 * Optimized version
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

// Global variable used by to count how many blocks have finished reduction
__device__ unsigned int reducedBlockCount = 0;

__global__ void
setImageByte( byte* dst, int stride, int x, int y, byte val)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row == x & col == y)
    dst[row * stride + col] = val;

}

// Extracting connected components using 8 neighbors
// and intersection with original image
__global__ void
dilate8IntersectionImageByte(byte* dst, byte* src, byte *org, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  byte res;

  // Dilate morphological operation
  if ( (src[(row - 1) * width + col - 1] == 255) |
	   (src[(row - 1) * width + col] == 255) |
	   (src[(row - 1) * width + col + 1] == 255) |
       (src[row * width + col - 1] == 255) |
       (src[row * width + col + 1] == 255) |
       (src[(row + 1) * width + col - 1] == 255) |
       (src[(row + 1) * width + col] == 255) |
       (src[(row + 1) * width + col + 1] == 255)
       )
  {
	  res = 255;
  }
  else
  {
	  res = src[row * width + col];
  }

  // Intersection with original
  dst[row * width + col] = res & org[row * width + col];
}

// Extracting connected components using 4 neighbors
// and intersection with original image
__global__ void
dilate4IntersectionImageByte(byte* dst, byte* src, byte *org, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  byte res;

  // Dilate morphological operation
  if ( (src[(row - 1) * width + col] == 255) |
       (src[row * width + col - 1] == 255) |
       (src[row * width + col + 1] == 255) |
       (src[(row + 1) * width + col] == 255) )
  {
	  res = 255;
  }
  else
  {
	  res = src[row * width + col];
  }

  // Intersection with original
  dst[row * width + col] = res & org[row * width + col];
}

// NOT USED
__global__ void
diffImageSimple( byte* dst, byte* xk, byte *xk1, int stride)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  dst[row * stride + col] = abs(xk[row * stride + col] - xk1[row * stride + col]);

}

__global__ void
compareImageSimple( byte* dst, byte* xk, byte *xk1, int stride)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (abs(xk[row * stride + col] - xk1[row * stride + col]) > 0)
	  dst[0] = 255;
}

// NOT USED for test only
__global__ void
diffImageReductionOpt( byte* dst, byte* xk, byte *xk1, int stride)
{
  __shared__ byte partialDiff[BLOCK_SIZE][BLOCK_SIZE];

  unsigned int ty = threadIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + ty;
  unsigned int col = blockIdx.x * blockDim.x + tx;
  unsigned int strideX, strideY;

  // Read block into shared memory - coalesced
  partialDiff[tx][ty] = abs(xk[row * stride + col] - xk1[row * stride + col]);
  __syncthreads();

  // Reduction for current block - in shared device memory
  for (strideX = blockDim.x >> 1; strideX > 0; strideX >>= 1)
  {
	  for (strideY = blockDim.y >> 1; strideY > 0; strideY >>= 1)
	  {
		  __syncthreads();
		  if (tx < strideX & ty < strideY)
			  partialDiff[tx][ty] |= partialDiff[tx+strideX][ty+strideY];
	  }
  }

  // Ensure all threads in block completed and write result to device memory
  __syncthreads();
  dst[row * stride + col] = partialDiff[tx][ty];
  __syncthreads();

  // First thread (0,0) in any block
  if (tx == 0 & ty == 0)
  {
	 // Atomic increment for thread (0,0) in any block
     //atomicAdd(&reducedBlockCount, 1);

     // The last block sums the results of all other blocks
     if( reducedBlockCount == (gridDim.x * gridDim.y) )
     {
    	 byte res = 0;

    	 // Sum result from all blocks
    	 for (row = 0; row < gridDim.x*blockDim.x; row += blockDim.x)
    		 for (col = 0; col < gridDim.y*blockDim.y; col += blockDim.y)
    			 res |= dst[row * stride + col];

    	 // Store final result
    	 dst[0] = res;

    	 // reset block count so that next run succeeds
    	 reducedBlockCount = 0;
     }
  }

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
	bool found = false;

	// Search first pixel with value >= val
	for (int x = 0; x < Size.height; x++) {
		for (int y = 0; y < Size.width; y++) {
			if (img[x*Stride + y] >= val) {
				point->x = x;
				point->y = y;
				found = true;
				break;
			}
		}
	}
	return found;
}

inline void swapPtr(byte **p1, byte **p2)
{
	byte *tmp = *p1;
	*p1 = *p2;
	*p2 = tmp;
}

// NOT USED for test only
float TestReduceImage(byte *dst, byte *imgA, ROI Size, int Stride)
{
	int ImgResStride;
	size_t ImgDevStride;
	byte *ImgDevA;
	byte *ImgDevB;
    byte *ImgDevTmp;

    DEBUG_MSG("[TestReduceImage]\n");

    byte *ImgTmp = MallocPlaneByte(Size.width, Size.height, &ImgResStride);

    // Create image A on device
    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevA), &ImgDevStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMemcpy2D(ImgDevA, ImgDevStride * sizeof(byte),
    						   imgA, Stride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );

    // Create image B on device
    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevB), &ImgDevStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMemcpy2D(ImgDevB, ImgDevStride * sizeof(byte),
    						   imgA, Stride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );

    cutilSafeCall(cudaMallocPitch((void **)(&ImgDevTmp), &ImgDevStride, Size.width * sizeof(byte), Size.height));

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)Size.width / BLOCK_SIZE), ceil((float)Size.height / BLOCK_SIZE) );

    DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    // Not equal
   	//setImageByte<<< grid, threads >>>(ImgDevB, ImgDevStride, 10, 10, 255); // Set arbitrary points
   	//setImageByte<<< grid, threads >>>(ImgDevB, ImgDevStride, 200, 200, 255); // Set arbitrary points
   	setImageByte<<< grid, threads >>>(ImgDevB, ImgDevStride, 287, 351, 255); // Set arbitrary points - error ?
   	//setImageByte<<< grid, threads >>>(ImgDevB, ImgDevStride, Size.height-2, Size.width-2, 255); // Set arbitrary points

    // start CUDA timer
    if (timerCUDA == 0) CreateTimer(&timerCUDA);
    RestartTimer(timerCUDA);
    diffImageReductionOpt<<< grid, threads >>>(ImgDevTmp, ImgDevA, ImgDevB, ImgDevStride);
    StopTimer(timerCUDA);

    cutilSafeCall(cudaMemcpy2D(ImgTmp, ImgResStride * sizeof(byte),
    						   ImgDevTmp, ImgDevStride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyDeviceToHost) );

    cutilSafeCall(cudaMemcpy(ImgTmp, ImgDevTmp, 4, cudaMemcpyDeviceToHost));

    if (*ImgTmp == 0)
    	DEBUG_MSG("Image are equal\n");
    else
    	DEBUG_MSG("Image are not equal\n");

    cutilSafeCall(cudaMemcpy2D(dst, Stride * sizeof(byte),
    						   ImgTmp, ImgResStride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToHost) );

    cutilSafeCall(cudaFree(ImgDevA));
    cutilSafeCall(cudaFree(ImgDevB));
    cutilSafeCall(cudaFree(ImgDevTmp));
    FreePlane(ImgTmp);

	return GetTimer(timerCUDA);

}

float LabelObjects(byte *dst, byte *bw, ROI Size, int Stride, int *Numbers, int neighbors)
{
	int n = 1;
	POINT point;
	int ImgResStride;
	size_t ImgDevStride;
	byte *ImgDevXk1;
	byte *ImgDevXk;
    byte *ImgDevA;
    byte *ImgDevTmp;
    byte *ImgDevXres;

    DEBUG_MSG("[LabelObjects]\n");

    // Create result image on host
    byte *ImgTmp = MallocPlaneByte(Size.width, Size.height, &ImgResStride);
    byte *ImgXres = MallocPlaneByte(Size.width, Size.height, &ImgResStride);
    ImgResStride /= sizeof(byte);
    //DEBUG_MSG("ImgResStride %d\n", ImgResStride);

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
    //DEBUG_MSG("ImgDevStride %d\n", ImgDevStride);

    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)Size.width / BLOCK_SIZE), ceil((float)Size.height / BLOCK_SIZE) );

    DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

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
    		if (neighbors == 8)
    			dilate8IntersectionImageByte<<< grid, threads >>>(ImgDevXk, ImgDevXk1, ImgDevA, ImgDevStride);
    		else
    			dilate4IntersectionImageByte<<< grid, threads >>>(ImgDevXk, ImgDevXk1, ImgDevA, ImgDevStride);

       		//diffImageReductionOpt<<< grid, threads >>>(ImgDevTmp, ImgDevXk, ImgDevXk1, ImgDevStride);
            cutilSafeCall(cudaMemset2D(ImgDevTmp, ImgDevStride, 0, 4, 1));
    		compareImageSimple<<< grid, threads >>>(ImgDevTmp, ImgDevXk, ImgDevXk1, ImgDevStride);
    	    cutilSafeCall(cudaMemcpy(ImgTmp, ImgDevTmp, 4, cudaMemcpyDeviceToHost));

      	    if (*ImgTmp == 0)
      	    {

    	    	//DEBUG_MSG("Object %d found\n", n);
         	    // Images are equal
    	    	// h = find(Xk == 1); Xres(h) = n;
    	    	lableImageObject<<< grid, threads >>>(ImgDevXres, ImgDevXk, ImgDevStride, n);
    	    	n = n + 1;
    	    	break;
    	    }

  		    // Xk_1 = Xk, swap pointers, instead of copy images
      	    swapPtr(&ImgDevXk1, &ImgDevXk);
    	}

		// Copy result to host
	    cutilSafeCall(cudaMemcpy2D(ImgXres, ImgResStride * sizeof(byte),
	    						   ImgDevXres, ImgDevStride * sizeof(byte),
	                               Size.width * sizeof(byte), Size.height,
	                               cudaMemcpyDeviceToHost) );

    }

    StopTimer(timerCUDA);

    cutilCheckMsg("Kernel execution failed");

    DEBUG_MSG("Objects found                     : %d\n", (n - 1));

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

	*Numbers = n;
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




