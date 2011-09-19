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
#include "imageMorph_kernel.cu"

/**
*  The dimension of pixels block 8x8 
*/
#define BLOCK_SIZE			8

int g_TotalFailures = 0;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
template <class T>
void runTest( int argc, char** argv, int len);

template<class T>
void
computeGold( T* reference, T* idata, const unsigned int len) 
{
    const T T_len = static_cast<T>( len);
    for( unsigned int i = 0; i < len; ++i) 
    {
        reference[i] = idata[i] * T_len;
    }
}

// Loads image from file
// Allocates memory for source and destination of image 
// based size of image, image type must be bmp
int 
loadImage(char* fileName, const char* path, byte** imgSrc, byte** imgDst, ROI* imgSize, int *imgStride)
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
    *imgDst = MallocPlaneByte(ImgWidth, ImgHeight, imgStride);

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

    char ImageFname[] = "rice.bmp";
    //char ImageFname[] = "ricebw.bmp";
    char EdgeImageFname[] = "riceEdge.bmp";

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
    if (loadImage(ImageFname, argv[0], &ImgSrc, &ImgDst, &ImgSize, &ImgStride))
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
    
    /* Template test
    printf("> runTest<float,32>\n");
    runTest<float>( argc, argv, 32);
    printf("> runTest<int,512>\n");
    runTest<int>( argc, argv, 512);

    printf("\n[imageMorph] -> Test Results: %d Failures\n", g_TotalFailures);
    printf( (g_TotalFailures == 0) ? "PASSED\n" : "FAILED\n" );
    */
    
    //release byte planes
    FreePlane(ImgSrc);
    FreePlane(ImgDst);

    cutilExit(argc, argv);
}

// To completely templatize runTest (below) with cutil, we need to use 
// template specialization to wrap up CUTIL's array comparison and file writing
// functions for different types.  

// Here's the generic wrapper for cutCompare*
template<class T>
class ArrayComparator
{
public:
    CUTBoolean compare( const T* reference, T* data, unsigned int len)
    {
        fprintf(stderr, "Error: no comparison function implemented for this type\n");
        return CUTFalse;
    }
};

// Here's the specialization for ints:
template<>
class ArrayComparator<int>
{
public:
    CUTBoolean compare( const int* reference, int* data, unsigned int len)
    {
        return cutComparei(reference, data, len);
    }
};

// Here's the specialization for floats:
template<>
class ArrayComparator<float>
{
public:
    CUTBoolean compare( const float* reference, float* data, unsigned int len)
    {
        return cutComparef(reference, data, len);
    }
};

// Here's the generic wrapper for cutWriteFile*
template<class T>
class ArrayFileWriter
{
public:
    CUTBoolean write(const char* filename, T* data, unsigned int len, float epsilon)
    {
        fprintf(stderr, "Error: no file write function implemented for this type\n");
        return CUTFalse;
    }
};

// Here's the specialization for ints:
template<>
class ArrayFileWriter<int>
{
public:
    CUTBoolean write(const char* filename, int* data, unsigned int len, float epsilon)
    {
        return cutWriteFilei(filename, data, len, epsilon != 0);
    }
};

// Here's the specialization for floats:
template<>
class ArrayFileWriter<float>
{
public:
    CUTBoolean write(const char* filename, float* data, unsigned int len, float epsilon)
    {
        return cutWriteFilef(filename, data, len, epsilon);
    }
};


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
template<class T>
void
runTest( int argc, char** argv, int len) 
{
    int devID;
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
    printf("CUDA device [%s] has %d Multi-Processors [len %d]\n", deviceProps.name, deviceProps.multiProcessorCount, len);

    unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    unsigned int num_threads = len;
    unsigned int mem_size = sizeof( float) * num_threads;

    // allocate host memory
    T* h_idata = (T*) malloc( mem_size);
    // initalize the memory
    for( unsigned int i = 0; i < num_threads; ++i) 
    {
        h_idata[i] = (T) i;
    }

    // allocate device memory
    T* d_idata;
    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    // copy host memory to device
    cutilSafeCall( cudaMemcpy( d_idata, h_idata, mem_size,
                                cudaMemcpyHostToDevice) );

    // allocate device memory for result
    T* d_odata;
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));

    // setup execution parameters
    dim3  grid( 1, 1, 1);
    dim3  threads( num_threads, 1, 1);

    // execute the kernel
    testKernel<T><<< grid, threads, mem_size >>>( d_idata, d_odata);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // allocate mem for the result on host side
    T* h_odata = (T*) malloc( mem_size);
    // copy result from device to host
    cutilSafeCall( cudaMemcpy( h_odata, d_odata, sizeof(T) * num_threads,
                                cudaMemcpyDeviceToHost) );

    cutilCheckError( cutStopTimer( timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

    // compute reference solution
    T* reference = (T*) malloc( mem_size);
    computeGold<T>( reference, h_idata, num_threads);

    ArrayComparator<T> comparator;
    ArrayFileWriter<T> writer;

    // check result
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        cutilCheckError( writer.write( "./data/regression.dat",
                                     h_odata, num_threads, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        CUTBoolean res = comparator.compare( reference, h_odata, num_threads);
        printf( "Compare %s\n\n", (1 == res) ? "OK" : "MISMATCH");
        g_TotalFailures += (1 != res);
    }

    // cleanup memory
    free( h_idata);
    free( h_odata);
    free( reference);
    cutilSafeCall(cudaFree(d_idata));
    cutilSafeCall(cudaFree(d_odata));

    cudaThreadExit();
}
