/*
 * CensusDisparity.cu
 *
 * Computes the depth map based on the census algorithm
 * input is the left and right stereo image in BW
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

#define CALLOC calloc
#define COUNT_TABLE_BITS 256

static unsigned int timerCUDA = 0;

static void calc_table(int N, long int start, long int end, unsigned char *table, unsigned char count) {

  /*printf("%d\t%d\t%d\t%d\n",N,start,end,count);*/
  if (N == 1) {
    *(table + start) = count;

  } else {
    calc_table (N/2, start, start + (end-start+1)/2 - 1, table, count);
    calc_table (N/2, start + (end-start+1)/2, end, table, count + 1);
  }
} /* calc_table */

static void print_table(unsigned char *count_table, int N) {
  int i;
  for (i=0; i<N; i++)
    printf("%d\t%d\n",i,*(count_table+i));
}

static float census_transform(unsigned char *image, int x_window_size, int y_window_size, int width, int height, int num_buffs, int size_buff, long int *census_tx) {
  int i, j, x_surround, y_surround, top, bottom, left, right, x, y, incr, index, k;
  unsigned char *image_row_ptr, *pix_ptr, *top_corner, centre_val;
  long int *census_ptr;

  printf ("CensusTransform [%d,%d], [%d,%d], %d, %d\n",x_window_size, y_window_size, width, height, num_buffs, size_buff);

  if (timerCUDA == 0) CreateTimer(&timerCUDA);
  RestartTimer(timerCUDA);

  x_surround = (x_window_size - 1) / 2;
  y_surround = (y_window_size - 1) / 2;
  top = y_surround;
  left = x_surround;
  right = width - x_surround;
  bottom = height - y_surround;
  incr = width - x_window_size;

  image_row_ptr = image;

  for (y = top; y < bottom; y++) {
    census_ptr = census_tx + (y * width + left) * num_buffs;
    top_corner = image_row_ptr;

    for (x = left; x < right; x++) {
      pix_ptr =  top_corner;
      centre_val = *(top_corner + width * y_surround + x_surround);

      /* initialise census transform to 0 */
      for (i = 0; i < num_buffs; i++)
	    *(census_ptr + i) = 0;

      k = 0;
      for (i = 0; i < y_window_size; i++) {
		for (j = 0; j < x_window_size; j++) {
		  index = k / size_buff;
		  *(census_ptr + index) <<= 1;
		  if (*pix_ptr < centre_val)
			*(census_ptr + index) |= 1;
		  pix_ptr++;
		  k++;
		} /* for j */
		pix_ptr += incr;
      } /* for i */

      top_corner++;
      census_ptr += num_buffs;
    } /* for x */

    image_row_ptr += width;
  } /* for */

  StopTimer(timerCUDA);
  return GetTimer(timerCUDA);

} /* census_transform */


__global__ void
averageTest( long int* census, byte* src, int censusStride, int width, int num_buffs, int x_win_size, int y_win_size, int size_buff)
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

  // Average operation
  long int sum = (pix01 + pix10 + pix11 + pix12 + pix21)/5;

  census[row * censusStride * num_buffs + col] = sum;
}

// optimized version using shared memmory!!! see matrixmul_device.cu, not completely working and not faster!!!
__global__ void
censusTransformShare( long int* census, byte* src, int censusStride, int width, int num_buffs, int x_win_size, int y_win_size, int size_buff)
{
  // Using shared memory
  __shared__ byte pixels[BLOCK_SIZE][BLOCK_SIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int i, j, k, index;

  // Top left offset of census window
  int top = (x_win_size - 1) / 2;
  int left = (y_win_size - 1) / 2;
  int incr = width - x_win_size;

  // Center pixel value in census window
  byte centre_val = src[row * width + col];

  // Pixel pointer to top left corner of census window
  byte *pix_ptr = src + ((row - top) * width + col - left);

  // Pointer to census transform buffers
  long int *census_ptr = census + (row * censusStride * num_buffs + col);
  // Initialize census transform buffer to 0
  for (i = 0; i < num_buffs; i++) census_ptr[i] = 0;

  // Performs census transform of window size
  k = 0;
  for (i = 0; i < y_win_size; i++) {
    pixels[ty][tx] = *pix_ptr;
   __syncthreads();

    // x_win_size must be < BLOCK_SIZE
	for (j = 0; j < x_win_size; j++) {
	  index = k / size_buff;
	  *(census_ptr + index) <<= 1;
	  if (pixels[j][tx] < centre_val)
		*(census_ptr + index) |= 1;
	  pix_ptr++;
	  k++;
	}
	pix_ptr += incr;
    __syncthreads();
  }
}

__global__ void
censusTransform( long int* census, byte* src, int censusStride, int width, int num_buffs, int x_win_size, int y_win_size, int size_buff)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int i, j, k, index;

  // Top left offset of census window
  int top = (x_win_size - 1) / 2;
  int left = (y_win_size - 1) / 2;
  int incr = width - x_win_size;

  // Center pixel value in census window
  byte centre_val = src[row * width + col];

  // Pixel pointer to top left corner of census window
  byte *pix_ptr = src + ((row - top) * width + col - left);

  // Pointer to census transform buffers
  long int *census_ptr = census + (row * censusStride * num_buffs + col);
  // Initialize census transform buffer to 0
  for (i = 0; i < num_buffs; i++) census_ptr[i] = 0;

  // Performs census transform of window size
  k = 0;
  for (i = 0; i < y_win_size; i++) {
	for (j = 0; j < x_win_size; j++) {
	  index = k / size_buff;
	  *(census_ptr + index) <<= 1;
	  if (*pix_ptr < centre_val)
		*(census_ptr + index) |= 1;
	  pix_ptr++;
	  k++;
	}
	pix_ptr += incr;
  }
}

#define BUFF_SIZE (num_buffs * sizeof(long int))
#define PIXEL_SIZE sizeof(byte)
#define ADD_BOARDER(ptr, width) (ptr + width*BOARDER_SIZE + BOARDER_SIZE*BUFF_SIZE);

static float census_transform_cuda (unsigned char *image, int x_window_size, int y_window_size, int width, int height,
		                            int num_buffs, int size_buff, long int *census_tx)
{
    byte *SrcImg;
    long int *CensusTrans, *CensusTransB;
    size_t SrcStride, CensusStride, CensusWidth;
    ROI SB; // Size of image with and without black boarder
    int BOARDER_SIZE = (x_window_size - 1)/2; // Boarder size

    SB.width = width + BOARDER_SIZE*2; // Add black boarders to allocated device image memory buffers
    SB.height = height + BOARDER_SIZE*2;

    DEBUG_MSG("CensusTransform [%d,%d], [%d,%d], %d, %d\n", width, height, x_window_size, y_window_size, num_buffs, size_buff);

    if (y_window_size > x_window_size)
    	  printf("Census y_window_size [%d] must less or equal to x_window_size [%d]\n", y_window_size, x_window_size);

    if (timerCUDA == 0) CreateTimer(&timerCUDA);
    RestartTimer(timerCUDA);

    cutilSafeCall(cudaMallocPitch((void **)(&CensusTransB), &CensusStride, SB.width * BUFF_SIZE, SB.height));
    //DEBUG_MSG("CensusStride %d\n", CensusStride);
    CensusWidth = CensusStride/sizeof(long int);
    //DEBUG_MSG("CensusWidth %d\n", CensusWidth);

	cutilSafeCall(cudaMallocPitch((void **)(&SrcImg), &SrcStride, width * PIXEL_SIZE, height));
    //DEBUG_MSG("SrcStride %d\n", SrcStride);

    //copy source image from host memory to device
    cutilSafeCall(cudaMemcpy2D(SrcImg, SrcStride,
                               image, width * PIXEL_SIZE,
                               width * PIXEL_SIZE, height,
                               cudaMemcpyHostToDevice) );

    CensusTrans = ADD_BOARDER(CensusTransB, CensusWidth);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)width / BLOCK_SIZE), ceil((float)height / BLOCK_SIZE) );

    //DEBUG_MSG("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    //DEBUG_MSG("Threads in Block [%d,%d]\n", threads.x, threads.y);

    DEBUG_MSG("kernel censusTransform [%d,%d], [%d,%d], %d, %d\n", CensusWidth, SrcStride, x_window_size, y_window_size, num_buffs, size_buff);
    // Performs census transform using data parallel computing NVIDIA graphics card
    censusTransform<<< grid, threads >>>(CensusTrans, SrcImg, CensusWidth, SrcStride, num_buffs, x_window_size, y_window_size, size_buff);
    //censusTransformShare<<< grid, threads >>>(CensusTrans, SrcImg, CensusWidth, SrcStride, num_buffs, x_window_size, y_window_size, size_buff);

    // Copy census transform from device memory to host memory
    cutilSafeCall(cudaMemcpy2D(census_tx, width * BUFF_SIZE,
    		                    CensusTrans, CensusStride,
                                width * BUFF_SIZE, height,
                                cudaMemcpyDeviceToHost) );
    //clean up memory
    cutilSafeCall(cudaFree(CensusTransB));
    cutilSafeCall(cudaFree(SrcImg));

    StopTimer(timerCUDA);
    return GetTimer(timerCUDA);

}

float CENSUS_RIGHT_CUDA (unsigned char *left_image, unsigned char *right_image, signed char *disparity, double *min_array,
		           int width, int height, int x_census_win_size, int y_census_win_size, int x_window_size, int y_window_size, int min_disparity, int max_disparity) {
  unsigned int right_x;
  int right_lim, left_lim, y, i, top, bottom, left, right, x_surround, y_surround, diff, num_buffs, extra_bits, size_buff, div_buffs, u, v, incr, x_surr1, y_surr1;
  long int *census_left, *census_right, *ptr_censusl, *ptr_censusr, census_l, census_r, *buff_r, *buff_l, *lptr, *rptr, xor_res;
  int disp;
  float timeCUDA, totalTime = 0;

  unsigned char *count_table;

  DEBUG_MSG("CENSUS_RIGHT_CUDA\n");

  count_table = (unsigned char*) CALLOC(256, sizeof(unsigned char));
  calc_table (COUNT_TABLE_BITS, 0, COUNT_TABLE_BITS-1, count_table, 0);
  /* print_table(count_table,COUNT_TABLE_BITS); */

  size_buff = sizeof(long int) * 8; // 32
  div_buffs = (x_census_win_size * y_census_win_size) / size_buff;
  extra_bits = (x_census_win_size * y_census_win_size) % size_buff;
  num_buffs = div_buffs + ((extra_bits > 0)?1:0);

  buff_l = (long int*) CALLOC(num_buffs, sizeof(long int));
  buff_r = (long int*) CALLOC(num_buffs, sizeof(long int));

  census_left = (long int*) CALLOC(width * height * num_buffs, sizeof(long int));
  timeCUDA = census_transform_cuda (left_image, x_census_win_size, y_census_win_size, width, height, num_buffs, size_buff, census_left);
  // Doesn't seem to be faster at all using CUDA in this case ?????
  //timeCUDA = census_transform (left_image, x_census_win_size, y_census_win_size, width, height, num_buffs, size_buff, census_left);
  printf("Processing time of left image census transform (cuda)  : %f ms \n", timeCUDA);
  totalTime += timeCUDA;

  census_right = (long int*) CALLOC(width * height * num_buffs, sizeof(long int));
  timeCUDA = census_transform_cuda (right_image, x_census_win_size, y_census_win_size, width, height, num_buffs, size_buff, census_right);
  //timeCUDA = census_transform (right_image, x_census_win_size, y_census_win_size, width, height, num_buffs, size_buff, census_right);
  printf("Processing time of right image census transform (cuda) : %f ms \n", timeCUDA);
  totalTime += timeCUDA;

  x_surround = (x_window_size - 1) / 2;
  y_surround = (y_window_size - 1) / 2;
  x_surr1 = x_surround + x_census_win_size/2;
  y_surr1 = y_surround + y_census_win_size/2;
  top = y_surr1;
  left = x_surr1;
  right = width - x_surr1;
  bottom = height - y_surr1;
  incr = (width - x_window_size) * num_buffs;

  /* Set minimum array to a really large number */
  for (i = 0; i < width * height; i++)
    min_array[i] = 1E10;

  for (disp = min_disparity; disp < max_disparity; disp++) {

	//printf ("Disparity %d\n",disp);

    for (y = top; y < bottom; y++) {

      if (disp < 0) {
		ptr_censusl =  census_left + ((y - y_surround) * width + x_surr1 - x_surround) * num_buffs;
		ptr_censusr = census_right + ((y - y_surround) * width - disp + x_surr1 - x_surround) * num_buffs;

      } else {
		ptr_censusl =  census_left + ((y - y_surround) * width + disp + x_surr1 - x_surround) * num_buffs;
		ptr_censusr = census_right + ((y - y_surround) * width + x_surr1 - x_surround) * num_buffs;
      }

      right_lim = (disp < 0)? right : right - disp;
      left_lim = (disp < 0)? left - disp : left;
      /*printf("%d\n",y);*/

      for (right_x = left_lim; right_x < right_lim; right_x++) {

		lptr = ptr_censusl;
		rptr = ptr_censusr;

		diff = 0;
		for (u = 0; u < y_window_size; u++) {
		  for (v = 0; v < x_window_size * num_buffs; v++) {

			census_l = *lptr;
			census_r = *rptr;

			xor_res = census_l ^ census_r;
			for (i = 0; i < sizeof(long int); i++) {
			  diff += *(count_table + (xor_res & 0x00ff));
			}

			lptr ++;
			rptr ++;
		  } /* for v */

		  lptr += incr;
		  rptr += incr;
		} /* for u */

		if (diff < *(min_array + width * y + right_x)) {
		  *(disparity + width * y + right_x) = (unsigned char) disp; /* - min_disparity; */
		  *(min_array + width * y + right_x) = diff;
		} /* if */

		ptr_censusl += num_buffs;
		ptr_censusr += num_buffs;
      } /* for right_x */
    } /* for y*/
  } /* for disparity */

  free (count_table);
  free (buff_l); free(buff_r);
  free (census_left); free(census_right);

  printf("CUDA Census Right completed\n");

  return totalTime;
} /* CENSUS_RIGHT */

__global__ static void
average3DImages (byte* dst, int stride, cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
	  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	  byte* imgPtr = (byte *)devPitchedPtr.ptr;
	  size_t pitch = devPitchedPtr.pitch;
	  size_t slicePitch = pitch * height;
	  byte pixel[DEPTH];

	  for (int z = 0; z < depth; ++z)
	  {
		  byte *slice = imgPtr + z * slicePitch; // Find sliced image
		  byte *row = slice + rowIdx * pitch; // Find row in image
		  pixel[z] = row[colIdx]; // Create array with pixel in both images
	  }

	  // Update average of stereo images
	  dst[rowIdx * stride + colIdx] = (pixel[0] + pixel[1])/2;
	  //dst[rowIdx * stride + colIdx] = pixel[1]; // 0 = left image, 1 = rigth image
}

__device__ static void swap (byte *x, byte *y)
{
	byte tmp;
	tmp = *x;
	*x = *y;
	*y = tmp;
}

__device__ static void bublesort (byte *a, int depth)
{
	int i, j;
	for (i = 0; i < (depth-1); i++)
		for (j = 0; j < (depth-(i+1)); j++)
			if (a[j] > a[j+1])
				swap(&a[j], &a[j+1]);
}


__global__ static void
median3DImages (byte* dst, int stride, cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
	  int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
	  int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
	  byte* imgPtr = (byte *)devPitchedPtr.ptr;
	  size_t pitch = devPitchedPtr.pitch;
	  size_t slicePitch = pitch * height;
	  byte median[DEPTH];

	  for (int z = 0; z < depth; ++z)
	  {
		  byte *slice = imgPtr + z * slicePitch; // Find sliced image
		  byte *row = slice + rowIdx * pitch; // Find row in image
		  median[z] = row[colIdx];
	  }

	  bublesort(median, depth);

	  dst[rowIdx * stride + colIdx] = median[(DEPTH+1)/2];
}

// Find depth map image based on 3D cube of images representing left and right images
float CensusDisparity(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride, int depth,
		              int x_census_win_size, int y_census_win_size, int x_window_size, int y_window_size, int min_disparity, int max_disparity)
{
    byte *Dst;
    size_t DstStride;
    cudaMemcpy3DParms memcpy3DParms = {0};

    DEBUG_MSG("[CensusDisparity]\n");

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

    //median3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);
    average3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);

    StopTimer(timerCUDA);

    cutilSafeCall(cudaThreadSynchronize());

    cutilSafeCall(cudaMemcpy2D(ImgDst, Size.width * sizeof(byte),
                                Dst, DstStride * sizeof(byte),
                                Size.width * sizeof(byte), Size.height,
                                cudaMemcpyDeviceToHost) );

    cutilSafeCall(cudaFree(memcpy3DParms.dstPtr.ptr));

    return GetTimer(timerCUDA);;
}



