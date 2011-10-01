/*
 * imageThrust.h
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

#ifndef IMAGETHRUST_H_
#define IMAGETHRUST_H_

// Computes difference between 2 images using the thrust library
float ThrustImageDiff(byte *ImgDst, byte *ImgBack, byte *ImgSrc, ROI Size, int ISStride, int IBStride);

#endif /* IMAGETHRUST_H_ */
