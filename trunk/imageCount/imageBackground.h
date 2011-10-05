/*
 * imageBackground.h
 *
 * Finding background in a series of images in 3D, where the z-dimension is the time
 * the background image is found computing the median of the pixel intensity in the z-dimension
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

#ifndef IMAGEBACKGROUND_H_
#define IMAGEBACKGROUND_H_

// Find background image based on 3D cube of images
float ImageBackground(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride, int depth);

#endif /* IMAGEBACKGROUND_H_ */
