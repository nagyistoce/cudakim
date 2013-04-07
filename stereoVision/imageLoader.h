/*
 * imageLoader.h
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

#ifndef IMAGELOADER_H_
#define IMAGELOADER_H_

extern "C"
{

	// Moved image pointer to next image
	byte *NextImage(byte *pImage, int imgStride, ROI size);

	// Loads image from file
	// Allocates memory for source and destination of image
	// based size of image, image type must be bmp
	int loadImage(char* fileName, const char* path, byte** imgSrc, ROI* imgSize, int *imgStride);

	// Loads image from files 1-9
	// Allocates memory for source images in 3D cube
	// based size of image, image type must be bmp
	int loadImages(char* fileName, const char* path, byte** imgSrc, ROI* imgSize, int *imgStride, int depth);

}

#endif /* IMAGELOADER_H_ */
