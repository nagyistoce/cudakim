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

// includes, kernels
#include "image_kernel.cu"

/**
*  The dimension of pixels block 16x16
*/
#define BLOCK_SIZE			16

int g_TotalFailures = 0;


byte *NextImage(byte *pImage, int stepBytes, ROI size)
{
	return (pImage + (stepBytes*size.height));
}

// Loads image from file
// Allocates memory for source and destination of image 
// based size of image, image type must be bmp
int 
loadImage(char* fileName, const char* path, byte** imgSrc, byte** imgDst, ROI* imgSize, int *imgStride, int depth)
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
    *imgDst = MallocPlaneByte(ImgWidth, ImgHeight, imgStride);

    imgSize->width = ImgWidth;
    imgSize->height = ImgHeight;
    imgCur = *imgSrc;

    //load sample images
    for (int i = 1; i <= depth; i++)
    {
        printf("Loading image %s \n", ImageName);
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

float MorphEdgeCUDA1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
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
    unsigned int timerCUDA = 0;
    cutilCheckError(cutCreateTimer(&timerCUDA));
    cutilCheckError(cutResetTimer(timerCUDA));
    cutilCheckError(cutStartTimer(timerCUDA));
    
    //copy image from device memory to device memory
    /*
    cutilSafeCall(cudaMemcpy2D(Dst, DstStride * sizeof(float),  
                                Src, SrcStride * sizeof(float), 
                                Size.width * sizeof(float), Size.height,
                                cudaMemcpyDeviceToDevice) );
    
    copyImage<<< grid, threads >>>(Dst, Src, Size.width);
    */

    // Generate BW image
    tresholdImage<<< grid, threads >>>(DstBW, Src, Size.width, 110);
    cutilSafeCall(cudaThreadSynchronize());

    // Dilate image with structuring element
    dilateImage<<< grid, threads >>>(Dst, DstBW, Size.width);
    // Erode image with structuring element
    //erodeImage<<< grid, threads >>>(Dst, DstBW, Size.width);
    cutilSafeCall(cudaThreadSynchronize());
    
    // Diff BW and eroded image
    diffImage<<< grid, threads >>>(Diff, DstBW, Dst, Size.width);
    cutilSafeCall(cudaThreadSynchronize());

    cutilCheckError(cutStopTimer(timerCUDA));

    cutilCheckMsg("Kernel execution failed");

    // finalize CUDA timer
    float TimerCUDASpan = cutGetAverageTimerValue(timerCUDA);
    cutilCheckError(cutDeleteTimer(timerCUDA));

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
    return TimerCUDASpan;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	byte *ImgSrc, *ImgDst;
	ROI ImgSize;
	int ImgStride;
    printf("[imageMorph]\n");
    int devID;

    //char ImageFname[] = "rice.bmp";
    //char ImageFname[] = "ricebw.bmp";
    char ImageFname[] = "data/E45nord%d.bmp";
    char EdgeImageFname[] = "nordEdge.bmp";

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
    
    // Load image and allocate memory
    if (loadImage(ImageFname, argv[0], &ImgSrc, &ImgDst, &ImgSize, &ImgStride, 9))
    {
        //finalize
        cutilExit(argc, argv);
        return 1;
    }
    
    // Test image - Rice black/white image
    printf("Image 0[0,%d], 10[0,%d], 11[255,%d], 22[255,%d], 23[0,%d], 256[0,%d]\n", 
            ImgSrc[256], ImgSrc[256*9], ImgSrc[256*10], ImgSrc[256*21], ImgSrc[256*22], ImgSrc[256*255]);

    printf("Erode image\n");
    float TimeCUDA1 = MorphEdgeCUDA1(ImgSrc, ImgDst, ImgStride, ImgSize);
    printf("Processing time (ErodeCUDA 1)    : %f ms \n", TimeCUDA1);
    
    //dump result of Gold 1 processing
    printf("Success\nDumping result to %s...\n", EdgeImageFname);
    DumpBmpAsGray(EdgeImageFname, ImgDst, ImgStride, ImgSize);

    //release byte planes
    FreePlane(ImgSrc);
    FreePlane(ImgDst);

    cutilExit(argc, argv);
}
