/*
 * This program counts objects in a series of images
 * the algorithm reads 9 images specified by ImageFName
 * the background is extracted and moving objects are
 * isolated, located and labeled.
 *
 * The resulting images are colored and stored in 9 files.
 *
 * The background image is stored in: BackImageFname
 * The result image is stored in: TestImageFname
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
#include "imageLoader.h"
#include "deviceUtil.h"

// includes, kernels and functions
#include "imageBackground.h"

static unsigned int timerTotalCUDA = 0;

/* Remaining work list:
*
* OK - Color result images
* OK - Update input images using Matlab - remove header time
* OK - Matlab tic toc measure time
* OK - Optimize labelObjects - reduction kernel (Err)
* OK - Run computeprof
*
* - Using ideas from guest lecture Allan Rasmusson, connected components analysis
* - Optimize dilation using tile and shared memory
* - Gausian blurring of foreground images
*/

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	byte *ImgSrc, *ImgDst, *ImgDiff, *ImgBW, *ImgBack, *ImgCur;
	ROI ImgSize;
	int ImgSrcStride, ImgDstStride, ImgBackStride, ImgBWStride;
    int devID;
    int depth = DEPTH;
    float TimeCUDA;
    float TimeTotal = 0;
    float TimeLableObjects = 0;
    char ImageName[50];
    int ObjectsFound, Objects;

    char ImageFname[] = "data/Bowling-%d.bmp";
    char BackImageFname[] = "bowlingBackground.bmp";
    char TestImageFname[] = "bowlingResult%d.bmp";

    printf("StereoVision version 1.0\n");
    printf("Program performs depth map computation based on stereo images\n");
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

    // Load images 1-9 and allocate memory
    if (loadImages(ImageFname, argv[0], &ImgSrc, &ImgSize, &ImgSrcStride, depth))
    {
        //finalize
        cutilExit(argc, argv);
        return 1;
    }
    
    // Initialize timer
    CreateTimer(&timerTotalCUDA);
    StartTimer(timerTotalCUDA);

    ImgBack = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgBackStride);

    TimeCUDA = ImageBackground(ImgBack, ImgSrc, ImgSize, ImgSrcStride, depth);
    printf("Processing time (ImageBackground)    : %f ms \n", TimeCUDA);
    TimeTotal += TimeCUDA;

    // Save temporary background image in file
    //Dump result of finding background image
    printf("Dumping background image to %s...\n", BackImageFname);
    DumpBmpAsGray(BackImageFname, ImgBack, ImgBackStride, ImgSize);
    //------------------------------------------------------------------------------------------
/*
    // Allocate images
    ImgDst = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgDstStride);
    ImgDiff = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgBWStride);
    ImgBW = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgBWStride);

    printf("--------------------------------------------------\n");
    printf("Locating and label of objects based on background \n");

    ImgCur = ImgSrc;
    ObjectsFound = 0;

    for (int i = 1; i <= depth; i++)
    {
		TimeCUDA = DiffImages(ImgDiff, ImgBack, ImgCur, ImgSize, ImgSrcStride, ImgBackStride);
		// Experimental implementation of diff images using thrust
		//TimeCUDA = ThrustImageDiff(ImgDst, ImgBack, ImgCur, ImgSize, ImgSrcStride, ImgBackStride);
		printf("Processing time (DiffImages)      : %f ms \n", TimeCUDA);
	    TimeTotal += TimeCUDA;

	    ImgCur = NextImage(ImgCur, ImgSrcStride, ImgSize);

		TimeCUDA = MorphObjects(ImgBW, ImgDiff, ImgSize, ImgBWStride);
	    printf("Processing time (MorphObjects)    : %f ms \n", TimeCUDA);
	    TimeTotal += TimeCUDA;

	    TimeCUDA = LabelObjects(ImgDst, ImgBW, ImgSize, ImgDstStride, &Objects, 8); // Using 8 or 4 connected neighbors
	    //TimeCUDA = TestReduceImage(ImgDst, ImgBW, ImgSize, ImgDstStride); // Test reduce kernel
	    printf("Processing time (LabelObjects)    : %f ms \n", TimeCUDA);
	    TimeTotal += TimeCUDA;
	    TimeLableObjects += TimeCUDA;

	    ObjectsFound += Objects;

		sprintf(ImageName, TestImageFname, i);
		printf("Dumping BW image to %s...\n", ImageName);
		DumpBmpAsGray(ImageName, ImgBW, ImgBWStride, ImgSize);
		//printf("Dumping Diff image to %s...\n", ImageName);
		//DumpBmpAsGray(ImageName, ImgDiff, ImgBWStride, ImgSize);

		sprintf(ImageName, TestImageFname, i+10);
		printf("Dumping Label image to %s...\n", ImageName);
		DumpBmpColorMap(ImageName, ImgDst, ImgDstStride, ImgSize, redColorMap, RED_COLOR_MAP_SIZE);
		//DumpBmpAsGray(ImageName, ImgDst, ImgDstStride, ImgSize);
    }
*/
    StopTimer(timerTotalCUDA);
    float time = GetTimer(timerTotalCUDA);
    printf("Image processing time (Total)     : %f ms \n", TimeTotal);
    printf("Image label objects time (Total)  : %f ms \n", TimeLableObjects);
    printf("Processing time (Total)           : %f ms \n", time);

    //printf("Total number of objects found     : %d \n", ObjectsFound);

    //release byte planes
    FreePlane(ImgSrc);
 	FreePlane(ImgBack);
 /*
    FreePlane(ImgDst);
    FreePlane(ImgDiff);
    FreePlane(ImgBW);
*/
    cutilExit(argc, argv);
}
