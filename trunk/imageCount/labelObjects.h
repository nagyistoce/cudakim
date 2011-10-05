/*
 * labelObjects.h
 *
 * Finds connected components in binary (BW) image
 * using 8 or 4 neighbor pixels
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

#ifndef LABELOBJECTS_H_
#define LABELOBJECTS_H_

// Label objects found in image bw and stores result in dst image
float LabelObjects(byte *dst, byte *bw, ROI Size, int Stride, int *Numbers, int neighbors);

// Test of reduce algorithm for comparing images (NOT USED)
float TestReduceImage(byte *dst, byte *imgA, ROI Size, int Stride);

#endif /* LABELOBJECTS_H_ */
