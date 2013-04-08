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
*  Modified on: 26/09/2011
*           by: kim bjerge
*/

/**
**************************************************************************
* \file BmpUtil.cpp
* \brief Contains basic image operations implementation.
*
* This file contains implementation of basic bitmap loading, saving, 
* conversions to different representations and memory management routines.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cutil_inline.h>

#include "defs.h"
#include "BmpUtil.h"


#ifdef _WIN32
#pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

BMPColorMap redColorMap[RED_COLOR_MAP_SIZE] =
{     //r,   g,   b      map index
	{	0,   0,   0 },  // 0
	{ 255,  20,   0 },  // Red
	{  60, 255, 220 },
	{ 255, 220,   0 },
	{ 255,  40,   0 },
	{  80, 255, 200 },
	{ 255, 200,   0 },
	{ 255,  60,   0 },
	{ 100, 255, 180 },
	{ 255,  80,   0 },
	{ 255, 180,   0 },
	{ 120, 255, 160 },
	{ 220, 255,  60 },
	{ 255, 100,   0 },
	{ 140, 255, 140 },
	{ 255, 120,   0 },  // Orange
	{ 160, 255, 120 },  // Green
	{ 240, 255,  40 },
	{ 255, 140,   0 },
	{ 180, 255, 100 },
	{ 255, 255,  20 },
	{ 255, 160,   0 },
	{ 200, 255,  80 },
	{ 255, 240,   0 },  // Yellow
	{  50, 255, 255 }   // Blue
};

/**
**************************************************************************
*  The routine clamps the input value to integer byte range [0, 255]
*
* \param x			[IN] - Input value
*  
* \return Pointer to the created plane
*/
int clamp_0_255(int x)
{
	return (x < 0) ? 0 : ( (x > 255) ? 255 : x );
}


/**
**************************************************************************
*  Float round to nearest value
*
* \param num			[IN] - Float value to round
*  
* \return The closest to the input float integer value
*/
float round_f(float num) 
{
	float NumAbs = fabs(num);
	int NumAbsI = (int)(NumAbs + 0.5f);
	float sign = num > 0 ? 1.0f : -1.0f;
	return sign * NumAbsI;
}


/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 8bpp pixels.
*
* \param width			[IN] - Width of image buffer to be allocated
* \param height			[IN] - Height of image buffer to be allocated
* \param pStepBytes		[OUT] - Step between two sequential rows
*  
* \return Pointer to the created plane
*/
byte *MallocPlaneByte(int width, int height, int *pStepBytes)
{
	byte *ptr;
	*pStepBytes = ((int)ceil(width/16.0f))*16;
	ptr = (byte *)malloc(*pStepBytes * height);
	return ptr;
}

/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 8bpp pixels.
*
* \param width			[IN] - Width of image buffer to be allocated
* \param height			[IN] - Height of image buffer to be allocated
* \param depth			[IN] - Depth of image buffer to be allocated
* \param pStepBytes		[OUT] - Step between two sequential rows
*
* \return Pointer to the created plane
*/
byte *MallocCubeByte(int width, int height, int depth, int *pStepBytes)
{
	byte *ptr;
	*pStepBytes = ((int)ceil(width/16.0f))*16;
	ptr = (byte *)malloc(*pStepBytes * height * depth);
	return ptr;
}

/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 16bpp float pixels.
*
* \param width			[IN] - Width of image buffer to be allocated
* \param height			[IN] - Height of image buffer to be allocated
* \param pStepBytes		[OUT] - Step between two sequential rows
*  
* \return Pointer to the created plane
*/
short *MallocPlaneShort(int width, int height, int *pStepBytes)
{
	short *ptr;
	*pStepBytes = ((int)ceil((width*sizeof(short))/16.0f))*16;
	ptr = (short *)malloc(*pStepBytes * height);
	*pStepBytes = *pStepBytes / sizeof(short);
	return ptr;
}


/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 32bpp float pixels.
*
* \param width			[IN] - Width of image buffer to be allocated
* \param height			[IN] - Height of image buffer to be allocated
* \param pStepBytes		[OUT] - Step between two sequential rows
*  
* \return Pointer to the created plane
*/
float *MallocPlaneFloat(int width, int height, int *pStepBytes)
{
	float *ptr;
	*pStepBytes = ((int)ceil((width*sizeof(float))/16.0f))*16;
	ptr = (float *)malloc(*pStepBytes * height);
	*pStepBytes = *pStepBytes / sizeof(float);
	return ptr;
}


/**
**************************************************************************
*  Copies byte plane to float plane
*
* \param ImgSrc				[IN] - Source byte plane
* \param StrideB			[IN] - Source plane stride
* \param ImgDst				[OUT] - Destination float plane
* \param StrideF			[IN] - Destination plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void CopyByte2Float(byte *ImgSrc, int StrideB, float *ImgDst, int StrideF, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgDst[i*StrideF+j] = (float)ImgSrc[i*StrideB+j];
		}
	}
}


/**
**************************************************************************
*  Copies float plane to byte plane (with clamp)
*
* \param ImgSrc				[IN] - Source float plane
* \param StrideF			[IN] - Source plane stride
* \param ImgDst				[OUT] - Destination byte plane
* \param StrideB			[IN] - Destination plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void CopyFloat2Byte(float *ImgSrc, int StrideF, byte *ImgDst, int StrideB, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgDst[i*StrideB+j] = (byte)clamp_0_255((int)(round_f(ImgSrc[i*StrideF+j])));
		}
	}
}


/**
**************************************************************************
*  Memory deallocator, deletes aligned format frame.
*
* \param ptr			[IN] - Pointer to the plane
*  
* \return None
*/
void FreePlane(void *ptr)
{
	if (ptr) 
	{
		free(ptr);
	}
}


/**
**************************************************************************
*  Performs addition of given value to each pixel in the plane
*
* \param Value				[IN] - Value to add
* \param ImgSrcDst			[IN/OUT] - Source float plane
* \param StrideF			[IN] - Source plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void AddFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgSrcDst[i*StrideF+j] += Value;
		}
	}
}


/**
**************************************************************************
*  Performs multiplication of given value with each pixel in the plane
*
* \param Value				[IN] - Value for multiplication
* \param ImgSrcDst			[IN/OUT] - Source float plane
* \param StrideF			[IN] - Source plane stride
* \param Size				[IN] - Size of area to copy
*  
* \return None
*/
void MulFloatPlane(float Value, float *ImgSrcDst, int StrideF, ROI Size)
{
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			ImgSrcDst[i*StrideF+j] *= Value;
		}
	}
}


/**
**************************************************************************
*  This function performs acquisition of image dimensions
*
* \param FileName		[IN] - Image name to load
* \param Width			[OUT] - Image width from file header
* \param Height			[OUT] - Image height from file header
*
* \return Status code
*/
int PreLoadBmp(char *FileName, int *Width, int *Height)
{
	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;
	FILE *fh;
	
	if (!(fh = fopen(FileName, "rb")))
	{
		return 1; //invalid filename
	}

	fread(&FileHeader, sizeof(BMPFileHeader), 1, fh);

	if (FileHeader._bm_signature != 0x4D42)
	{
		return 2; //invalid file format
	}

	fread(&InfoHeader, sizeof(BMPInfoHeader), 1, fh);

	if (InfoHeader._bm_color_depth != 24)
	{
		return 3; //invalid color depth
	}

	if(InfoHeader._bm_compressed)
	{
		return 4; //invalid compression property
	}

	*Width  = InfoHeader._bm_image_width;
	*Height = InfoHeader._bm_image_height;

	fclose(fh);
	return 0;
}


/**
**************************************************************************
*  This function performs loading of bitmap luma
*
* \param FileName		[IN] - Image name to load
* \param Stride			[IN] - Image stride
* \param ImSize			[IN] - Image size
* \param Img			[OUT] - Prepared buffer
*
* \return None
*/
void LoadBmpAsGray(char *FileName, int Stride, ROI ImSize, byte *Img)
{
	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;
	FILE *fh;
	fh = fopen(FileName, "rb");

	fread(&FileHeader, sizeof(BMPFileHeader), 1, fh);
	fread(&InfoHeader, sizeof(BMPInfoHeader), 1, fh);

	for (int i=ImSize.height-1; i>=0; i--)
	{
		for (int j=0; j<ImSize.width; j++)
		{
			int r=0, g=0, b=0;
			fread(&b, 1, 1, fh);
			fread(&g, 1, 1, fh);
			fread(&r, 1, 1, fh);
			int val = ( 313524*r + 615514*g + 119537*b + 524288) >> 20 ;
			Img[i*Stride+j] = (byte)clamp_0_255(val);
		}
	}

	fclose(fh);
	return;
}

/**
**************************************************************************
*  This function performs dumping of bitmap luma on HDD
*
* \param FileName		[OUT] - Image name to dump to
* \param Img			[IN] - Image luma to dump
* \param Stride			[IN] - Image stride
* \param ImSize			[IN] - Image size
* \param Map			[IN] - Color map
* \param SizeMap		[IN] - Size of color map
*
* \return None
*/
void DumpBmpColorMap(char *FileName, byte *Img, int Stride, ROI ImSize, BMPColorMap *Map, int SizeMap)
{
	FILE *fp = NULL;
	fp = fopen(FileName, "wb");
    if (fp == NULL)
	{
    	return;
    }

	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;

	//init headers
	FileHeader._bm_signature = 0x4D42;
	FileHeader._bm_file_size = 54 + 3 * ImSize.width * ImSize.height;
	FileHeader._bm_reserved = 0;
	FileHeader._bm_bitmap_data = 0x36;
	InfoHeader._bm_bitmap_size = 0;
	InfoHeader._bm_color_depth = 24;
	InfoHeader._bm_compressed = 0;
	InfoHeader._bm_hor_resolution = 0;
	InfoHeader._bm_image_height = ImSize.height;
	InfoHeader._bm_image_width = ImSize.width;
	InfoHeader._bm_info_header_size = 40;
	InfoHeader._bm_num_colors_used = 0;
	InfoHeader._bm_num_important_colors = 0;
	InfoHeader._bm_num_of_planes = 1;
	InfoHeader._bm_ver_resolution = 0;

	fwrite(&FileHeader, sizeof(BMPFileHeader), 1, fp);
	fwrite(&InfoHeader, sizeof(BMPInfoHeader), 1, fp);

	for (int i = ImSize.height - 1; i>=0; i--)
	{
		for (int j=0; j<ImSize.width; j++)
		{
			byte mapIdx = Img[i*Stride+j];
			if (mapIdx >= SizeMap)
				mapIdx = SizeMap-1;
			fwrite(&(Map[mapIdx].b), 1, 1, fp);
			fwrite(&(Map[mapIdx].g), 1, 1, fp);
			fwrite(&(Map[mapIdx].r), 1, 1, fp);
		}
	}

	fclose(fp);
}


/**
**************************************************************************
*  This function performs dumping of bitmap luma on HDD 
*
* \param FileName		[OUT] - Image name to dump to
* \param Img			[IN] - Image luma to dump
* \param Stride			[IN] - Image stride
* \param ImSize			[IN] - Image size
*
* \return None
*/
void DumpBmpAsGray(char *FileName, byte *Img, int Stride, ROI ImSize)
{
	FILE *fp = NULL;
	fp = fopen(FileName, "wb");
    if (fp == NULL) 
	{
    	return;
    }

	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;

	//init headers
	FileHeader._bm_signature = 0x4D42;
	FileHeader._bm_file_size = 54 + 3 * ImSize.width * ImSize.height;
	FileHeader._bm_reserved = 0;
	FileHeader._bm_bitmap_data = 0x36;
	InfoHeader._bm_bitmap_size = 0;
	InfoHeader._bm_color_depth = 24;
	InfoHeader._bm_compressed = 0;
	InfoHeader._bm_hor_resolution = 0;
	InfoHeader._bm_image_height = ImSize.height;
	InfoHeader._bm_image_width = ImSize.width;
	InfoHeader._bm_info_header_size = 40;
	InfoHeader._bm_num_colors_used = 0;
	InfoHeader._bm_num_important_colors = 0;
	InfoHeader._bm_num_of_planes = 1;
	InfoHeader._bm_ver_resolution = 0;

	fwrite(&FileHeader, sizeof(BMPFileHeader), 1, fp);
	fwrite(&InfoHeader, sizeof(BMPInfoHeader), 1, fp);

	for (int i = ImSize.height - 1; i>=0; i--)
	{
		for (int j=0; j<ImSize.width; j++)
		{
			fwrite(&(Img[i*Stride+j]), 1, 1, fp);
			fwrite(&(Img[i*Stride+j]), 1, 1, fp);
			fwrite(&(Img[i*Stride+j]), 1, 1, fp);
		}
	}

	fclose(fp);
}

void DumpBmpAsGrayOffset(char *FileName, signed char *Img, int Stride, ROI ImSize, int offset)
{
	FILE *fp = NULL;
	fp = fopen(FileName, "wb");
    if (fp == NULL)
	{
    	return;
    }

	BMPFileHeader FileHeader;
	BMPInfoHeader InfoHeader;

	//init headers
	FileHeader._bm_signature = 0x4D42;
	FileHeader._bm_file_size = 54 + 3 * ImSize.width * ImSize.height;
	FileHeader._bm_reserved = 0;
	FileHeader._bm_bitmap_data = 0x36;
	InfoHeader._bm_bitmap_size = 0;
	InfoHeader._bm_color_depth = 24;
	InfoHeader._bm_compressed = 0;
	InfoHeader._bm_hor_resolution = 0;
	InfoHeader._bm_image_height = ImSize.height;
	InfoHeader._bm_image_width = ImSize.width;
	InfoHeader._bm_info_header_size = 40;
	InfoHeader._bm_num_colors_used = 0;
	InfoHeader._bm_num_important_colors = 0;
	InfoHeader._bm_num_of_planes = 1;
	InfoHeader._bm_ver_resolution = 0;

	fwrite(&FileHeader, sizeof(BMPFileHeader), 1, fp);
	fwrite(&InfoHeader, sizeof(BMPInfoHeader), 1, fp);

	for (int i = ImSize.height - 1; i>=0; i--)
	{
		for (int j=0; j<ImSize.width; j++)
		{
			signed char pixel = Img[i*Stride+j] + offset;
			fwrite(&pixel, 1, 1, fp);
			fwrite(&pixel, 1, 1, fp);
			fwrite(&pixel, 1, 1, fp);
		}
	}

	fclose(fp);
}


/**
**************************************************************************
*  This function performs dumping of 8x8 block from float plane
*
* \param PlaneF			[IN] - Image plane
* \param StrideF		[IN] - Image stride
* \param Fname			[OUT] - File name to dump to
*
* \return None
*/
void DumpBlockF(float *PlaneF, int StrideF, char *Fname)
{
	FILE *fp = fopen(Fname, "wb");
	for (int i=0; i<8; i++)
	{
		for (int j=0; j<8; j++)
		{
			fprintf(fp, "%.*f  ", 14, PlaneF[i*StrideF+j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


/**
**************************************************************************
*  This function performs dumping of 8x8 block from byte plane
*
* \param Plane			[IN] - Image plane
* \param Stride			[IN] - Image stride
* \param Fname			[OUT] - File name to dump to
*
* \return None
*/
void DumpBlock(byte *Plane, int Stride, char *Fname)
{
	FILE *fp = fopen(Fname, "wb");
	for (int i=0; i<8; i++)
	{
		for (int j=0; j<8; j++)
		{
			fprintf(fp, "%.3d  ", Plane[i*Stride+j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


/**
**************************************************************************
*  This function performs evaluation of Mean Square Error between two images
*
* \param Img1			[IN] - Image 1
* \param Img2			[IN] - Image 2
* \param Stride			[IN] - Image stride
* \param Size			[IN] - Image size
*
* \return Mean Square Error between images
*/
float CalculateMSE(byte *Img1, byte *Img2, int Stride, ROI Size)
{
	uint32 Acc = 0;
	for (int i=0; i<Size.height; i++)
	{
		for (int j=0; j<Size.width; j++)
		{
			int TmpDiff = Img1[i*Stride+j] - Img2[i*Stride+j];
			TmpDiff	*= TmpDiff;
			Acc += TmpDiff;
		}
	}
	return ((float)Acc) / (Size.height * Size.width);
}


/**
**************************************************************************
*  This function performs evaluation of Peak Signal to Noise Ratio between 
*  two images
*
* \param Img1			[IN] - Image 1
* \param Img2			[IN] - Image 2
* \param Stride			[IN] - Image stride
* \param Size			[IN] - Image size
*
* \return Peak Signal to Noise Ratio between images
*/
float CalculatePSNR(byte *Img1, byte *Img2, int Stride, ROI Size)
{
	float MSE = CalculateMSE(Img1, Img2, Stride, Size);
	return 10 * log10(255*255 / MSE);
}
