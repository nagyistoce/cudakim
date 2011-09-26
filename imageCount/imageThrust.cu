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

// includes, thrust
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/adjacent_difference.h>
#include <thrust/random.h>

#include <iostream>
#include <iterator>

// includes, timer utilities
#include "defs.h"
#include "BmpUtil.h"
//#include "timer.h"

//static unsigned int timerCUDA = 0;

struct func_diff_byte
{
	__host__ __device__
	byte operator()(byte a, byte b)
	{
		return abs(a - b);
	}
};

// Performs thresholding and morphological operations like dilation and erode of image
// COULD BE OPTIMIZED! kbe???
float ThrustImageDiff(byte *ImgDst, byte *ImgBack, byte *ImgSrc, ROI Size, int ISStride, int IBStride)
{
    cudaEvent_t start;
    cudaEvent_t end;
    float elapsed_time;
    int idx, ImgSize = ISStride*Size.height*sizeof(byte);

    printf("[ThrustImageDiff]\n");

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    //if (timerCUDA == 0) CreateTimer(&timerCUDA);
    cudaEventRecord(start,0);

    thrust::host_vector<byte> hostImgBack(ImgSize);
    thrust::host_vector<byte> hostImgSrc(ImgSize);

    // How to copy ImgBack and ImgSrc to host_vectors efficient?
    for (idx = 0; idx < ImgSize; idx++)
    {
    	hostImgSrc[idx] = ImgSrc[idx];
    	hostImgBack[idx] = ImgBack[idx];
    }

    // Copy host vectors to devices
    thrust::device_vector<byte> devImgBack = hostImgBack;
    thrust::device_vector<byte> devImgSrc = hostImgSrc;
    thrust::device_vector<byte> devImgDst(ImgSize);

    //StartTimer(timerCUDA);
    func_diff_byte FuncDiff;
    thrust::transform(devImgSrc.begin(), devImgSrc.end(),
    		          devImgBack.begin(), devImgDst.begin(), FuncDiff);
    //StopTimer(timerCUDA);

    // transfer data back to host
    thrust::copy(devImgDst.begin(), devImgDst.end(), hostImgSrc.begin());

    // How to copy host_vectors to ImgDst efficient?
    for (idx = 0; idx < ImgSize; idx++)
    {
    	ImgDst[idx] = hostImgSrc[idx];
    }

    cudaThreadSynchronize();
    cudaEventRecord(end,0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, start, end);

    return (elapsed_time); // Total time
    //return(GetTimer(timerCUDA));
};
