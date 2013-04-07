/*
 * defs.h
 * General project definitions
 *
 *  Created on: 26/09/2011
 *      Author: kim bjerge
 *
 */

#ifndef DEFS_H_
#define DEFS_H_

/**
*  The dimension of threads in block 16x16
*/
#define BLOCK_SIZE			16

// Debug messages to printf
#define DEBUG_MSG			printf

// Debug messages do nothing
//#define DEBUG_MSG 			1 ? 0 :

/**
*  Number of images to analyze - should be equal to number of pictures to analyze
*/
#define DEPTH 				2 // Number of images to analyze (left+right)

typedef struct
{
	int x;				//!< x position
	int y;				//!< y position
} POINT;

/**
*  Simple 2D size / region_of_interest structure
*/
typedef struct
{
	int width;			//!< ROI width
	int height;			//!< ROI height
} ROI;

/**
*  One-byte unsigned integer type
*/
typedef unsigned char byte;

#endif /* DEFS_H_ */
