/*
 * CensusDisparity.h
 *
 * Computes the depth map based on the census algorithm
 * input is the left and right stereo image in BW
 *
 *  Created on: 8/4/2013
 *      Author: kimbjerge
 */

#ifndef CENSUSDISPARITY_H_
#define CENSUSDISPARITY_H_

// Computes disparity based on left and right images output the depth map in ImgDst
float CensusDisparity(byte *ImgDst, byte *ImgSrc, ROI Size, int Stride, int depth,
		              int x_census_win_size, int y_census_win_size, int x_window_size, int y_window_size, int min_disparity, int max_disparity);

void CENSUS_RIGHT (unsigned char *left_image, unsigned char *right_image, signed char *disparity, double *min_array, int width, int height,
		           int x_census_win_size, int y_census_win_size, int x_window_size, int y_window_size, int min_disparity, int max_disparity);
float CENSUS_RIGHT_CUDA (unsigned char *left_image, unsigned char *right_image, signed char *disparity, double *min_array, int width, int height,
		                int x_census_win_size, int y_census_win_size, int x_window_size, int y_window_size, int min_disparity, int max_disparity);

#endif /* CENSUSDISPARITY_H_ */
