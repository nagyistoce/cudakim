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
#include "censusDisparity.h"

// Settings for computing disparity
#define DEFAULT_WINDOW_SIZE    11
#define DEFAULT_TX_WINDOW_SIZE 5 // Census window size
//#define DEFAULT_TX_WINDOW_SIZE 7
#define DEFAULT_MAX_DISPARITY  100
#define DEFAULT_MIN_DISPARITY  0
#define MIN_ALLOWED_DISPARITY  -128
#define MAX_ALLOWED_DISPARITY  127

static unsigned int timerTotalCUDA = 0;

/* Remaining work list:
*
* - Gausian blurring of foreground images
*/

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	byte *ImgSrc, *ImgDst, *ImgDiff, *ImgBW, *ImgBack, *ImgRight, *ImgDepth;
	signed char *ImgDepthRef;
  	double *scores;
	ROI ImgSize;
	int ImgSrcStride, ImgDstStride, ImgBackStride, ImgDepthStride, ImgBWStride;
    int devID;
    int depth = DEPTH;
    float TimeCUDA;
    float TimeTotal = 0;
    float TimeLableObjects = 0;
    char ImageName[50];
    int ObjectsFound, Objects;

    char ImageFname[] = "data/BowlingC-%d.bmp";
    //char ImageFname[] = "data/E45Nord%d.bmp";
    char ImageLeftBWFname[] = "imageLeftBW.bmp";
    char ImageRightBWFname[] = "imageRightBW.bmp";
    char DepthTestFname[] = "imageDepthTest.bmp";
    char DepthMapFname[] = "imageDepthMap.bmp";
    char DepthMapRefFname[] = "imageDepthMapRef.bmp";
    char BackImageFname[] = "imageBackground.bmp";
    char TestImageFname[] = "imageResult%d.bmp";

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
    
    // Save left and right BW image in files
    printf("Dumping left and right BW images to %s, %s...\n", ImageLeftBWFname, ImageRightBWFname);
    DumpBmpAsGray(ImageLeftBWFname, ImgSrc, ImgSrcStride, ImgSize);
    ImgRight = NextImage(ImgSrc, ImgSrcStride, ImgSize);
    DumpBmpAsGray(ImageRightBWFname, ImgRight, ImgSrcStride, ImgSize);

    // Initialize timer
    CreateTimer(&timerTotalCUDA);
    StartTimer(timerTotalCUDA);

    //------------------------------------------------------------------------------------------
    // Background processing using median filtering
    //------------------------------------------------------------------------------------------
    ImgBack = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgBackStride);

    TimeCUDA = ImageBackground(ImgBack, ImgSrc, ImgSize, ImgSrcStride, depth);
    printf("Processing time (ImageBackground)    : %f ms \n", TimeCUDA);
    TimeTotal += TimeCUDA;
    // Save temporary background image in file
    // Dump result of finding background image
    DumpBmpAsGray(BackImageFname, ImgBack, ImgBackStride, ImgSize);

    //------------------------------------------------------------------------------------------
    // Stereo depth map processing using Census transformation in computing disparities
    //------------------------------------------------------------------------------------------
    {
    int max_disparity = DEFAULT_MAX_DISPARITY, min_disparity = DEFAULT_MIN_DISPARITY,
      x_window_size = DEFAULT_WINDOW_SIZE, y_window_size = DEFAULT_WINDOW_SIZE,
      x_tx_win_size = DEFAULT_TX_WINDOW_SIZE, y_tx_win_size = DEFAULT_TX_WINDOW_SIZE;

    ImgDepthRef = (signed char *)MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgDepthStride);
    ImgDepth = MallocPlaneByte(ImgSize.width, ImgSize.height, &ImgDepthStride);

    TimeCUDA = CensusDisparity(ImgDepth, ImgSrc, ImgSize, ImgDepthStride, depth,
    		                   x_tx_win_size, y_tx_win_size, x_window_size, y_window_size, min_disparity, max_disparity);
    printf("Processing time (Census depth map)    : %f ms \n", TimeCUDA);
    TimeTotal += TimeCUDA;
    // Save depth map image in file
    // Dump result of finding stereo depth map
    DumpBmpAsGray(DepthTestFname, ImgDepth, ImgDepthStride, ImgSize);

    // Create depth map using host processor as reference
    if ((scores = (double*) calloc (ImgSize.width * ImgSize.height, sizeof(double))) == NULL) {
        //finalize
        cutilExit(argc, argv);
        return 1;
    }

#if 0  // Creates reference "C" version of stereo vision depth map
    CENSUS_RIGHT(ImgSrc,  ImgRight, ImgDepthRef, scores, ImgSize.width, ImgSize.height,
     		     x_tx_win_size, y_tx_win_size, x_window_size, y_window_size, min_disparity, max_disparity);

    // Save reference depth map image in file
    // Dump result of finding stereo depth map
    DumpBmpAsGrayOffset(DepthMapRefFname, ImgDepthRef, ImgDepthStride, ImgSize, min_disparity);
#endif

    TimeCUDA = CENSUS_RIGHT_CUDA(ImgSrc,  ImgRight, ImgDepthRef, scores, ImgSize.width, ImgSize.height,
     		     x_tx_win_size, y_tx_win_size, x_window_size, y_window_size, min_disparity, max_disparity);
    TimeTotal += TimeCUDA;

    // Save cuda computed depth map image in file
    // Dump result of finding stereo depth map
    DumpBmpAsGrayOffset(DepthMapFname, ImgDepthRef, ImgDepthStride, ImgSize, min_disparity);

    }

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
    //printf("Image label objects time (Total)  : %f ms \n", TimeLableObjects);
    printf("Processing time (Total)           : %f ms \n", time);

    //printf("Total number of objects found     : %d \n", ObjectsFound);

    //release byte planes
    FreePlane(ImgSrc);
 	FreePlane(ImgBack);
 	FreePlane(ImgDepth);
 	FreePlane(ImgDepthRef);
    free (scores);

 /*
    FreePlane(ImgDst);
    FreePlane(ImgDiff);
    FreePlane(ImgBW);
*/
    cutilExit(argc, argv);
}
