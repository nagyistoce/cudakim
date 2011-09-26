/*
 * labelObjects.h
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

#ifndef LABELOBJECTS_H_
#define LABELOBJECTS_H_

// Label objects found in image bw and stores result in dst image
float LabelObjects(byte *dst, byte *bw, ROI Size, int Stride);

#endif /* LABELOBJECTS_H_ */
