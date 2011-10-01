/*
 * locateObjects.h
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

#ifndef LOCATEOBJECTS_H_
#define LOCATEOBJECTS_H_


float DiffImages(byte *ImgDst, byte *ImgBack, byte *ImgSrc, ROI Size, int ISStride, int IBStride);

// Performs thresholding and morphological operations like dilation and erode of image
float MorphObjects(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride);

// Performs thresholding and morphological operations like dilation and erode of image
float MorphObjectsFloat(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride);


#endif /* LOCATEOBJECTS_H_ */
