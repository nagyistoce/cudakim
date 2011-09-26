/*
 * imageLoader.cpp
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, bmp utilities
#include "defs.h"
#include "BmpUtil.h"
#include "imageLoader.h"

// Moved image pointer to next image
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

    printf("[loadImage]\n%s\n", pImageFpath);

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

    printf("[loadImages]\n");

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


    printf("Images size [%d * %d * %d], stride %d \n", ImgWidth, ImgHeight, depth, *imgStride);

    return 0;
}



