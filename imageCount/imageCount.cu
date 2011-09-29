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
#include "defs.h"
#include "BmpUtil.h"
#include "timer.h"
#include "imageLoader.h"
#include "deviceUtil.h"

// includes, kernels and functions
#include "imageBackground.h"
#include "locateObjects.h"
#include "labelObjects.h"
#include "imageThrust.h"

static unsigned int timerTotalCUDA = 0;

/* Remaining work -
* OK - Color result images
* OK - Update input images using matlab - remove header time
* - Optimize labelObjects - reduction kernel
* - Run computeprof
* - Gausian bluring of diff images
* - Document results
*/

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	byte *ImgSrc, *ImgDst, *ImgDiff, *ImgBW, *ImgBack, *ImgCur;
	ROI ImgSize, ImgBackSize;
	int ImgSrcStride, ImgDstStride, ImgBackStride, ImgBWStride;
    int devID;
    int depth = DEPTH;
    float TimeCUDA;
    char ImageName[50];
    int ObjectsFound, Objects;

    //char ImageFname[] = "rice.bmp";
    char ImageFname[] = "data/E45nord%d.bmp";
    char BackImageFname[] = "nordBackground.bmp";
    char ImageBackFname[] = "nordBack.bmp";
    char TestImageFname[] = "nordResult%d.bmp";

    printf("Program counting objects in series of images\n");
    printf("--------------------------------------------------\n");

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
    PrintDeviceProperties();

    // Initialize timer
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

    //printf("Image src stride %d\n", ImgSrcStride);

#if 0 // Save temporary background image in file
    TimeCUDA = ImageBackground(ImgDst, ImgSrc, ImgSize, ImgSrcStride, depth);
    printf("Processing time (ImageBackground)    : %f ms \n", TimeCUDA);

    //Dump result of finding background image
    printf("Dumping background image to %s...\n", BackImageFname);
    DumpBmpAsGray(BackImageFname, ImgDst, ImgDstStride, ImgSize);

    printf("--------------------------------------------------\n");

    //------------------------------------------------------------------------------------------
    // Testing of diff background with images
    // Load image and allocate memory
    if (loadImage(ImageBackFname, argv[0], &ImgBack, &ImgBackSize, &ImgBackStride))
    {
        //finalize
        cutilExit(argc, argv);
        return 1;
    }
#else
    TimeCUDA = ImageBackground(ImgBack, ImgSrc, ImgSize, ImgSrcStride, depth);
    printf("Processing time (ImageBackground)    : %f ms \n", TimeCUDA);
#endif

    // Allocate BW image
    ImgDiff = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgBWStride);
    ImgBW = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgBWStride);

    printf("--------------------------------------------------\n");
    printf("Locating and label of objects based on %s...\n", ImageBackFname);

    ImgCur = ImgSrc;
    ObjectsFound = 0;
    for (int i = 1; i <= depth; i++)
    {
		TimeCUDA = DiffImages(ImgDiff, ImgBack, ImgCur, ImgSize, ImgSrcStride, ImgBackStride);
		//TimeCUDA = ThrustImageDiff(ImgDst, ImgBack, ImgCur, ImgSize, ImgSrcStride, ImgBackStride);
		printf("Processing time (DiffImages)      : %f ms \n", TimeCUDA);
		ImgCur = NextImage(ImgCur, ImgSrcStride, ImgSize);

		TimeCUDA = MorphObjects(ImgBW, ImgDiff, ImgSize, ImgBWStride);
	    printf("Processing time (MorphObjects)    : %f ms \n", TimeCUDA);

	    TimeCUDA = LabelObjects(ImgDst, ImgBW, ImgSize, ImgDstStride, &Objects);
	    printf("Processing time (LabelObjects)    : %f ms \n", TimeCUDA);
	    ObjectsFound += Objects;

		sprintf(ImageName, TestImageFname, i);
		//printf("Dumping BW image to %s...\n", ImageName);
		//DumpBmpAsGray(ImageName, ImgBW, ImgBWStride, ImgSize);
		printf("Dumping Diff image to %s...\n", ImageName);
		DumpBmpAsGray(ImageName, ImgDiff, ImgBWStride, ImgSize);

		sprintf(ImageName, TestImageFname, i+10);
		printf("Dumping Label image to %s...\n", ImageName);
		//DumpBmpAsGray(ImageName, ImgDst, ImgDstStride, ImgSize);
		DumpBmpColorMap(ImageName, ImgDst, ImgDstStride, ImgSize, redColorMap, RED_COLOR_MAP_SIZE);
    }

    StopTimer(timerTotalCUDA);
    float time = GetTimer(timerTotalCUDA);
    printf("Processing time (Total)         : %f ms \n", time);
    printf("Total number of objects found   : %d \n", ObjectsFound);

    //release byte planes
    FreePlane(ImgSrc);
    FreePlane(ImgDst);
    FreePlane(ImgDiff);
    FreePlane(ImgBack);
    FreePlane(ImgBW);

    cutilExit(argc, argv);
}
