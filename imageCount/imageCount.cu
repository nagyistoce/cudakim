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
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil_inline.h>

// includes, bmp utilities
#include "BmpUtil.h"
#include "timer.h"

// includes, kernels
#include "image_kernel.cu"

// External defined function
float ThrustImageDiff(byte *ImgBack, byte *ImgSrc, byte *ImgDst, ROI Size, int ISStride, int IBStride);

/**
*  The dimension of pixels block 16x16
*/
#define BLOCK_SIZE			16

static unsigned int timerCUDA = 0;
static unsigned int timerTotalCUDA = 0;
int g_TotalFailures = 0;


byte *NextImage(byte *pImage, int imgStride, ROI size)
{
	return (pImage + (imgStride*size.height));
}

// Loads image from file
// Allocates memory for source and destination of image 
// based size of image, image type must be bmp
int 
loadImage(char* fileName, const char* path, byte** imgSrc, ROI* imgSize, int *imgStride)
{
    //preload image (acquire dimensions)
    int ImgWidth, ImgHeight;
    //char *pImageFpath = cutFindFilePath(fileName, path);
    char *pImageFpath = fileName;

    int res = PreLoadBmp(pImageFpath, &ImgWidth, &ImgHeight);
    imgSize->width = ImgWidth;
    imgSize->height = ImgHeight;

    if (res)
    {
        printf("\nError %d: Image file %s not found or invalid!\n", res, pImageFpath);
        printf("Press ENTER to exit...\n");
        getchar();

        return 1;
    }

    //allocate image buffers
    *imgSrc = MallocPlaneByte(ImgWidth, ImgHeight, imgStride);

    //load sample image
    LoadBmpAsGray(pImageFpath, *imgStride, *imgSize, *imgSrc);
    //check image dimensions are multiples of BLOCK_SIZE
    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
    {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        printf("Press ENTER to exit...\n");
        getchar();

        return 1;
    }

    printf("Image size [%d x %d], %d \n", ImgWidth, ImgHeight, *imgStride);

    return 0;
}

// Loads image from files 1-9
// Allocates memory for source images in 3D cube
// based size of image, image type must be bmp
int
loadImages(char* fileName, const char* path, byte** imgSrc, ROI* imgSize, int *imgStride, int depth)
{
    //preload image (acquire dimensions)
    int ImgWidth, ImgHeight;
    byte *imgCur;
    //char *pImageFpath = cutFindFilePath(fileName, path);
    char ImageName[50];

    sprintf(ImageName, fileName, 1);

    int res = PreLoadBmp(ImageName, &ImgWidth, &ImgHeight);
    if (res)
    {
        printf("\nError %d: Image file %s not found or invalid!\n", res, ImageName);
        printf("Press ENTER to exit...\n");
        getchar();

        return 1;
    }

    //check image dimensions are multiples of BLOCK_SIZE
    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
    {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        printf("Press ENTER to exit...\n");
        getchar();

        return 1;
    }

    //allocate image buffers
    *imgSrc = MallocCubeByte(ImgWidth, ImgHeight, depth, imgStride);

    imgSize->width = ImgWidth;
    imgSize->height = ImgHeight;
    imgCur = *imgSrc;

    //load sample images
    for (int i = 1; i <= depth; i++)
    {
        printf("Loading image %s [%d,%d] \n", ImageName, ImgWidth, ImgHeight);
    	sprintf(ImageName, fileName, i);
    	res = PreLoadBmp(ImageName, &ImgWidth, &ImgHeight);
        if (res)
        {
            printf("\nError %d: Image file %s not found or invalid!\n", res, ImageName);
            printf("Press ENTER to exit...\n");
            getchar();
            return 1;
        }

    	LoadBmpAsGray(ImageName, *imgStride, *imgSize, imgCur);
    	imgCur = NextImage(imgCur, *imgStride, *imgSize);
    }

    
    printf("Images size [%d * %d * %d], %d \n", ImgWidth, ImgHeight, depth, *imgStride);
    
    return 0;
}

// NOT YET COMPLETED ! - kbe???
/*
float ImageBackgroundDiff(byte *ImgSrc, byte *ImgDst, ROI Size, int Stride, int depth)
{
    byte *Dst;
    size_t DstStride;
    cudaMemcpy3DParms memcpy3DParms = {0};

    // Create src pointer and extent
    memcpy3DParms.srcPtr = make_cudaPitchedPtr(ImgSrc, Stride, Size.width, Size.height);
    memcpy3DParms.extent = make_cudaExtent(Size.width * sizeof(byte), Size.height, depth);

    // Allocation of memory for 3D source images in byte format
    cutilSafeCall(cudaMalloc3D(&memcpy3DParms.dstPtr, memcpy3DParms.extent));

    printf("srcPtr: pitch, xsize, ysize [%d,%d,%d]\n", memcpy3DParms.srcPtr.pitch, memcpy3DParms.srcPtr.xsize, memcpy3DParms.srcPtr.ysize);
    printf("dstPtr: pitch, xsize, ysize [%d,%d,%d]\n", memcpy3DParms.dstPtr.pitch, memcpy3DParms.dstPtr.xsize, memcpy3DParms.dstPtr.ysize);

    // Copy images to device memory
    memcpy3DParms.kind = cudaMemcpyHostToDevice;
    cutilSafeCall(cudaMemcpy3D(&memcpy3DParms));

    // Allocation of memory for 2D destination image in byte format
    cutilSafeCall(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(byte), Size.height));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)Size.width / BLOCK_SIZE), ceil((float)Size.height / BLOCK_SIZE) );

    printf("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    printf("Threads in Block [%d,%d]\n", threads.x, threads.y);

    RestartTimer(timerCUDA);
    median3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);
    StopTimer(timerCUDA);
    float time = GetTimer(timerCUDA);
    //test3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);
    cutilSafeCall(cudaThreadSynchronize());

    //diff3DImages

    printf("Copy result to host\n");
    cutilSafeCall(cudaMemcpy2D(ImgDst, Size.width * sizeof(byte),
                                Dst, DstStride * sizeof(byte),
                                Size.width * sizeof(byte), Size.height,
                                cudaMemcpyDeviceToHost) );

    cutilSafeCall(cudaFree(memcpy3DParms.dstPtr.ptr));

    return time;
}
*/

// Find background image based on 3D cube of images
float ImageBackground(byte *ImgSrc, byte *ImgDst, ROI Size, int Stride, int depth)
{
    byte *Dst;
    size_t DstStride;
    cudaMemcpy3DParms memcpy3DParms = {0};

    // Create src pointer and extent
    memcpy3DParms.srcPtr = make_cudaPitchedPtr(ImgSrc, Stride, Size.width, Size.height);
    memcpy3DParms.extent = make_cudaExtent(Size.width * sizeof(byte), Size.height, depth);

    // Allocation of memory for 3D source images in byte format
    cutilSafeCall(cudaMalloc3D(&memcpy3DParms.dstPtr, memcpy3DParms.extent));

    printf("srcPtr: pitch, xsize, ysize [%d,%d,%d]\n", memcpy3DParms.srcPtr.pitch, memcpy3DParms.srcPtr.xsize, memcpy3DParms.srcPtr.ysize);
    printf("dstPtr: pitch, xsize, ysize [%d,%d,%d]\n", memcpy3DParms.dstPtr.pitch, memcpy3DParms.dstPtr.xsize, memcpy3DParms.dstPtr.ysize);

    // Copy images to device memory
    memcpy3DParms.kind = cudaMemcpyHostToDevice;
    cutilSafeCall(cudaMemcpy3D(&memcpy3DParms));

    // Allocation of memory for 2D destination image in byte format
    cutilSafeCall(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(byte), Size.height));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid( ceil((float)Size.width / BLOCK_SIZE), ceil((float)Size.height / BLOCK_SIZE) );

    printf("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    printf("Threads in Block [%d,%d]\n", threads.x, threads.y);

    RestartTimer(timerCUDA);
    median3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);
    //test3DImages<<< grid, threads >>>(Dst, DstStride, memcpy3DParms.dstPtr, Size.width, Size.height, depth);
    StopTimer(timerCUDA);
    float time = GetTimer(timerCUDA);
    cutilSafeCall(cudaThreadSynchronize());

    printf("Copy result to host\n");
    cutilSafeCall(cudaMemcpy2D(ImgDst, Size.width * sizeof(byte),
                                Dst, DstStride * sizeof(byte),
                                Size.width * sizeof(byte), Size.height,
                                cudaMemcpyDeviceToHost) );

    cutilSafeCall(cudaFree(memcpy3DParms.dstPtr.ptr));

    return time;
}

float ImageDiff(byte *ImgBack, byte *ImgSrc, byte *ImgDst, ROI Size, int ISStride, int IBStride)
{
    byte  *Diff, *Back, *Src;
    size_t DiffStride, SrcStride, BackStride;

    // Allocation of device memory for 2D difference image
    cutilSafeCall(cudaMallocPitch((void **)(&Diff), &DiffStride, Size.width * sizeof(byte), Size.height));
    DiffStride /= sizeof(byte);
    printf("DiffStride %d\n", DiffStride);

    // Allocation of memory for 2D background and source image in byte format
    cutilSafeCall(cudaMallocPitch((void **)(&Back), &BackStride, Size.width * sizeof(byte), Size.height));
    BackStride /= sizeof(byte);
    printf("BackStride %d\n", BackStride);

    cutilSafeCall(cudaMallocPitch((void **)(&Src), &SrcStride, Size.width * sizeof(byte), Size.height));
    SrcStride /= sizeof(byte);
    printf("SrcStride %d\n", SrcStride);

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

    printf("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    printf("Threads in Block [%d,%d]\n", threads.x, threads.y);

    RestartTimer(timerCUDA);
    diffImageByte<<< grid, threads >>>(Diff, Back, Src, SrcStride);
    StopTimer(timerCUDA);
    float time = GetTimer(timerCUDA);

    cutilSafeCall(cudaMemcpy2D(ImgDst, IBStride * sizeof(byte),
                                Diff, DiffStride * sizeof(byte),
                                Size.width * sizeof(byte), Size.height,
                                cudaMemcpyDeviceToHost) );

    //clean up memory
    cutilSafeCall(cudaFree(Diff));
    cutilSafeCall(cudaFree(Back));
    cutilSafeCall(cudaFree(Src));

    return time;
}

/* Matlab version
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

float LabelObjects(byte *dst, byte *bw, ROI Size, int stride, dim3 grid, dim3 threads)
{
	int x, y, k = 1, n = 2;
	bool found = false;
	int ImgResStride;

    byte *ImgRes = MallocPlaneByte(Size.width, Size.height, &ImgResStride);

	// Search white pixel part of
	for (x = 0; x < Size.height; x++) {
		for (y = 0; y < Size.width; y++) {
			byte pixel = bw[x*stride + y];
			if (pixel >= 255.0f) {
				found = true;
				break;
			}
		}
	}

	FreePlane(ImgRes);

	// Find first
	return 0;
}

// Performs thresholding and morphological operations like dilation and erode of image
float MorphObjects(byte *ImgSrc, byte *ImgDst, ROI Size, int Stride)
{    
    byte *Src, *DstBW, *Dst1, *Dst2;
    size_t DstStride, SrcStride;

    // Allocation of memory for 2D source image in single precision format
    cutilSafeCall(cudaMallocPitch((void **)(&Src), &SrcStride, Size.width * sizeof(byte), Size.height));
    SrcStride /= sizeof(byte);
    printf("SrcStride %d\n", SrcStride);

    //copy source image from host memory to device
    cutilSafeCall(cudaMemcpy2D(Src, SrcStride * sizeof(byte),
                               ImgSrc, Stride * sizeof(byte),
                               Size.width * sizeof(byte), Size.height,
                               cudaMemcpyHostToDevice) );

    // Allocation of device memory for 2D destination image in single precision format
    cutilSafeCall(cudaMallocPitch((void **)(&DstBW), &DstStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMallocPitch((void **)(&Dst1), &DstStride, Size.width * sizeof(byte), Size.height));
    cutilSafeCall(cudaMallocPitch((void **)(&Dst2), &DstStride, Size.width * sizeof(byte), Size.height));
    DstStride /= sizeof(byte);
    printf("DstStride %d\n", DstStride);
    
    //setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    printf("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    printf("Threads in Block [%d,%d]\n", threads.x, threads.y);

    //create and start CUDA timer
    RestartTimer(timerCUDA);
    
    // Generate BW image
    tresholdImageByte<<< grid, threads >>>(DstBW, Src, DstStride, 15);
    cutilSafeCall(cudaThreadSynchronize());

    // Erode image with structuring element
    erodeImageByte<<< grid, threads >>>(Dst1, DstBW, DstStride);
    // Dilate image with structuring element
    dilateImageByte<<< grid, threads >>>(Dst2, Dst1, DstStride);
    
    //
    //LabelObjects(Dst1, Dst2, Size, DstStride, grid, threads);

    StopTimer(timerCUDA);

    cutilCheckMsg("Kernel execution failed");

    //copy eroded image from device memory to host memory in Src
    cutilSafeCall(cudaMemcpy2D(ImgDst, Stride * sizeof(byte),
                                Dst2, DstStride * sizeof(byte),
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
float MorphObjectsFloat(byte *ImgSrc, byte *ImgDst, ROI Size, int Stride)
{
    float *Dst, *DstBW, *Src, *Diff;
    size_t DstStride, SrcStride, DiffStride;

    //convert source image to float representation
    int ImgSrcFStride;
    float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
    CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);

    // Allocation of memory for 2D source image in single precision format
    cutilSafeCall(cudaMallocPitch((void **)(&Src), &SrcStride, Size.width * sizeof(float), Size.height));
    SrcStride /= sizeof(float);
    printf("SrcStride %d\n", SrcStride);

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

    printf("Grid (Blocks)    [%d,%d]\n", grid.x, grid.y);
    printf("Threads in Block [%d,%d]\n", threads.x, threads.y);

    //create and start CUDA timer
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
    tresholdImage<<< grid, threads >>>(DstBW, Src, DstStride, 15);
    cutilSafeCall(cudaThreadSynchronize());

    // Erode image with structuring element
    erodeImage<<< grid, threads >>>(Dst, DstBW, DstStride);
    // Dilate image with structuring element
    dilateImage<<< grid, threads >>>(Diff, Dst, DstStride);
    //dilateImage<<< grid, threads >>>(Diff, DstBW, DstStride);
    cutilSafeCall(cudaThreadSynchronize());

    // Diff BW and eroded image
    //diffImage<<< grid, threads >>>(Diff, DstBW, Dst, Size.width);
    //cutilSafeCall(cudaThreadSynchronize());

    StopTimer(timerCUDA);
    float time = GetTimer(timerCUDA);

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
    return time;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	byte *ImgSrc, *ImgDst, *ImgBW, *ImgBack, *ImgCur;
	ROI ImgSize, ImgBackSize;
	int ImgSrcStride, ImgDstStride, ImgBackStride, ImgBWStride;
    int devID;
    int depth = DEPTH;
    float TimeCUDA;
    char ImageName[50];

    printf("[imageCount]\n");

    //char ImageFname[] = "rice.bmp";
    char ImageFname[] = "data/E45nord%d.bmp";
    char EdgeImageFname[] = "nordEdge.bmp";
    char ImageBackFname[] = "nordBack.bmp";
    char TestImageFname[] = "nordTest%d.bmp";

    cudaDeviceProp deviceProps;

	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
	    devID = cutilDeviceInit(argc, argv);
            if (devID < 0) {
               printf("exiting...\n");
               cutilExit(argc, argv);
               exit(0);
            }
	}
	else {
	    devID = cutGetMaxGflopsDeviceId();
	    cudaSetDevice( devID );
	}
		
    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

    // Initialize timer
	CreateTimer(&timerCUDA);
	CreateTimer(&timerTotalCUDA);
	StartTimer(timerTotalCUDA);

    // Load images 1-9 and allocate memory
    if (loadImages(ImageFname, argv[0], &ImgSrc, &ImgSize, &ImgSrcStride, depth))
    {
        //finalize
        cutilExit(argc, argv);
        return 1;
    }
    
    ImgDst = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgDstStride);

    // Test image - Rice black/white image
    //printf("Image 0[0,%d], 10[0,%d], 11[255,%d], 22[255,%d], 23[0,%d], 256[0,%d]\n",
    //        ImgSrc[256], ImgSrc[256*9], ImgSrc[256*10], ImgSrc[256*21], ImgSrc[256*22], ImgSrc[256*255]);

    printf("Image src stride %d\n", ImgSrcStride);
    TimeCUDA = ImageBackground(ImgSrc, ImgDst, ImgSize, ImgSrcStride, depth);
    //TimeCUDA = ImageBackgroundDiff(ImgSrc, ImgDst, ImgSize, ImgSrcStride, depth);
    printf("Processing time (Background)    : %f ms \n", TimeCUDA);
    
    //dump result of Gold 1 processing
    printf("Success\nDumping result to %s...\n", EdgeImageFname);
    DumpBmpAsGray(EdgeImageFname, ImgDst, ImgDstStride, ImgSize);

    //------------------------------------------------------------------------------------------
    // Testing of diff background with images
    // Load image and allocate memory
    if (loadImage(ImageBackFname, argv[0], &ImgBack, &ImgBackSize, &ImgBackStride))
    {
        //finalize
        cutilExit(argc, argv);
        return 1;
    }

    // Allocate BW image - result after thresholding and dilation
    ImgBW = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgBWStride);

    ImgCur = ImgSrc;
    for (int i = 1; i <= depth; i++)
    {
		TimeCUDA = ImageDiff(ImgBack, ImgCur, ImgDst, ImgSize, ImgSrcStride, ImgBackStride);
		//TimeCUDA = ThrustImageDiff(ImgBack, ImgCur, ImgDst, ImgSize, ImgSrcStride, ImgBackStride);
		printf("Processing time (Difference)    : %f ms \n", TimeCUDA);
		ImgCur = NextImage(ImgCur, ImgSrcStride, ImgSize);

		TimeCUDA = MorphObjects(ImgDst, ImgBW, ImgSize, ImgBWStride);
	    printf("Processing time (Morph)    : %f ms \n", TimeCUDA);

		sprintf(ImageName, TestImageFname, i);
		printf("Success\nDumping result to %s...\n", ImageName);
		//DumpBmpAsGray(ImageName, ImgDst, ImgDstStride, ImgSize);
		DumpBmpAsGray(ImageName, ImgBW, ImgBWStride, ImgSize);
    }

    StopTimer(timerTotalCUDA);
    float time = GetTimer(timerTotalCUDA);
    printf("Processing time (Total)    : %f ms \n", time);


    //release byte planes
    FreePlane(ImgSrc);
    FreePlane(ImgDst);
    FreePlane(ImgBack);
    FreePlane(ImgBW);

    cutilExit(argc, argv);
}
