/*
 * locateObjects.h
 *
 * Contains functions to isolate and locate foreground objects in image.
 * Includes functions to compute difference between images,
 * static image thresholding to produce a binary BW image,
 * performing morphological image operations like erode and dilate on BW images
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

#ifndef LOCATEOBJECTS_H_
#define LOCATEOBJECTS_H_

// Computes the difference between ImgBack and ImgSrc
float DiffImages(byte *ImgDst, byte *ImgBack, byte *ImgSrc, ROI Size, int ISStride, int IBStride);

// Performs thresholding and morphological operations like dilation and erode of image
float MorphObjects(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride);


#endif /* LOCATEOBJECTS_H_ */
