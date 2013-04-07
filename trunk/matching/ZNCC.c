/*
 * Copyright in this software is owned by CSIRO.  CSIRO grants permission to
 * any individual or institution to use, copy, modify, and distribute this
 * software, provided that:
 *
 * (a)     this copyright and permission notice appears in its entirety in or
 * on (as the case may be) all copies of the software and supporting
 * documentation;
 *
 * (b)     the authors of papers that describe software systems using this
 * software package acknowledge such use by citing the paper as follows:
 *
 *     "Quantitative Evaluation of Matching Methods and Validity Measures for
 *     Stereo Vision" by J. Banks and P. Corke, Int. J. Robotics Research,
 *     Vol 20(7), 2001; and
 *
 * (c)     users of this software acknowledge and agree that:
 *
 *   (i) CSIRO makes no representations about the suitability of this software
 *   for any purpose;
 *
 *   (ii) that the software is provided "as is" without express or implied
 *   warranty; and
 *
 *   (iii) users of this software use the software entirely at their own risk.
 *
 *  Author: Jasmine E. Banks (jbanks@ieee.org)
 */
#ifndef STANDALONE
#include "mex.h"
#define CALLOC mxCalloc

#else
#include <stdio.h>
#include <stdlib.h>
#include "area.h"
#define CALLOC calloc
#endif

#include <math.h>

/***************************************************************************************
compute_means
***************************************************************************************/
void 
compute_means_zncc(unsigned char *image_l, unsigned char *image_r, int x_window_size, int y_window_size, int width, int height, double *left_means, double *right_means)
{
	int             wx2, wy2, top, bottom, left, right, x, y, i, j,
	                incr, first_col;
	unsigned char  *right_corner, *left_corner, *left_row, *right_row,
	               *p_r, *p_l;
	double         *right_mean_ptr, *left_mean_ptr, size_sq;

	size_sq = x_window_size * y_window_size;
	wx2 = (x_window_size - 1) / 2;
	wy2 = (y_window_size - 1) / 2;
	top = wy2;
	left = wx2;
	right = width - wx2;
	bottom = height - wy2;
	incr = width - x_window_size;

	left_mean_ptr = left_means;
	right_mean_ptr = right_means;

	left_row = image_l;
	right_row = image_r;

	for (y = top; y < bottom; y++)
		for (x = left; x < right; x++) {
			*(right_means + y * width + x) = 0.0;
			*(left_means + y * width + x) = 0.0;

			for (i = -x_window_size / 2; i <= x_window_size / 2; i++)
				for (j = -y_window_size / 2; j <= y_window_size / 2; j++) {
					*(left_means + y * width + x) += (double) *(image_l + (y + j) * width + (x + i));
					*(right_means + y * width + x) += (double) *(image_r + (y + j) * width + (x + i));
				}

			*(right_means + y * width + x) /= size_sq;
			*(left_means + y * width + x) /= size_sq;
		}		/* for */
}				/* compute_means */

void 
match_ZNCC_right(unsigned char *image_l, unsigned char *image_r, signed char *disparity, double *max_array, int width, int height, int x_window_size, int y_window_size, int min_disparity, int max_disparity)
{
	unsigned int    right_x;
	unsigned char  *p_l, *p_r;
	int             i, j, y, top, bottom, left, right, wx2, wy2;
	double          pix_l, pix_r, *left_means, *right_means, *left_mean_ptr,
	               *right_mean_ptr;

	double          ncc, den;
	double          sum, sum_l, sum_r;
	int             disp, right_lim, left_lim;

	left_means = (double *) CALLOC(width * height, sizeof(double));
	right_means = (double *) CALLOC(width * height, sizeof(double));
	compute_means_zncc(image_l, image_r, x_window_size, y_window_size, width, height, left_means, right_means);

	wx2 = (x_window_size - 1) / 2;
	wy2 = (y_window_size - 1) / 2;
	top = wy2;
	left = wx2;
	right = width - wx2;
	bottom = height - wy2;

	for (i = 0; i < width * height; i++)
		max_array[i] = 0;

	for (disp = min_disparity; disp < max_disparity; disp++) {
#ifndef STANDALONE
		fprintf(stderr, "%d ", disp);
#else
		printf("%d\r", disp);
#endif

		for (y = top; y < bottom; y++) {
			if (disp < 0) {
				p_l = image_l + y * width + wx2;
				p_r = image_r + y * width - disp + wx2;
				left_mean_ptr = left_means + y * width + wx2;
				right_mean_ptr = right_means + y * width - disp + wx2;
			}
			else {
				p_l = image_l + y * width + disp + wx2;
				p_r = image_r + y * width + wx2;
				left_mean_ptr = left_means + y * width + disp + wx2;
				right_mean_ptr = right_means + y * width + wx2;
			}

			right_lim = (disp < 0) ? right : right - disp;
			left_lim = (disp < 0) ? left - disp : left;

			for (right_x = left_lim; right_x < right_lim; right_x++) {

				sum = 0;
				sum_l = 0;
				sum_r = 0;

				for (i = -wx2; i <= wx2; i++)
					for (j = -wy2; j <= wy2; j++) {
						pix_l = ((double) *(p_l + j * width + i)) - *left_mean_ptr;
						pix_r = ((double) *(p_r + j * width + i)) - *right_mean_ptr;

						sum += pix_l * pix_r;
						sum_l += pix_l * pix_l;
						sum_r += pix_r * pix_r;
					}	/* for j */

				den = sqrt(sum_l * sum_r);

				if (den != 0)
					ncc = (sum) / den;
				else
					ncc = 0;
				/* printf ("%lf ",ncc); */
				if (ncc > *(max_array + width * y + right_x)) {
					*(disparity + width * y + right_x) = disp;	/*- min_disparity;*/
					*(max_array + width * y + right_x) = ncc;
				}	/* if */

				p_l++;
				p_r++;
				left_mean_ptr++;
				right_mean_ptr++;
			}	/* for right_x */
		}		/* for y */
	}			/* for disparity */

#ifdef STANDALONE
	free(left_means);
	free(right_means);
#endif

	printf("\n");
}				/* match_ZNCC_right */

void 
match_ZNCC_left(unsigned char *image_l, unsigned char *image_r, signed char *disparity, double *max_array, int width, int height, int x_window_size, int y_window_size, int min_disparity, int max_disparity)
{
	unsigned int    left_x;
	int             i, j, y, top, bottom, left, right, wx2, wy2;
	unsigned char  *p_l, *p_r;
	double          pix_l, pix_r, *left_means, *right_means, *left_mean_ptr,
	               *right_mean_ptr;

	double          ncc, den;
	double          sum, sum_l, sum_r, prev_sum, prev_sum_l, prev_sum_r;
	int             disp, right_lim, left_lim;

	left_means = (double *) CALLOC(width * height, sizeof(double));
	right_means = (double *) CALLOC(width * height, sizeof(double));
	compute_means_zncc(image_l, image_r, x_window_size, y_window_size, width, height, left_means, right_means);

	wx2 = (x_window_size - 1) / 2;
	wy2 = (y_window_size - 1) / 2;
	top = wy2;
	left = wx2;
	right = width - wx2;
	bottom = height - wy2;

	for (i = 0; i < width * height; i++)
		max_array[i] = 0;

	for (disp = min_disparity; disp < max_disparity; disp++) {
#ifndef STANDALONE
		printf("%d ", disp);
#else
		printf("%d\r", disp);
#endif

		for (y = top; y < bottom; y++) {
			if (disp < 0) {
				p_l = image_l + y * width + wx2;
				p_r = image_r + y * width - disp + wx2;
				left_mean_ptr = left_means + y * width + wx2;
				right_mean_ptr = right_means + y * width - disp + wx2;
			}
			else {
				p_l = image_l + y * width + disp + wx2;
				p_r = image_r + y * width + wx2;
				left_mean_ptr = left_means + y * width + disp + wx2;
				right_mean_ptr = right_means + y * width + wx2;
			}

			right_lim = (disp < 0) ? right + disp : right;
			left_lim = (disp < 0) ? left : left + disp;

			for (left_x = left_lim; left_x < right_lim; left_x++) {

				sum = 0;
				sum_l = 0;
				sum_r = 0;

				for (i = -wx2; i < wx2; i++)
					for (j = -wy2; j < wy2; j++) {
						pix_l = ((double) *(p_l + j * width + i)) - *left_mean_ptr;
						pix_r = ((double) *(p_r + j * width + i)) - *right_mean_ptr;

						sum += pix_l * pix_r;
						sum_l += pix_l * pix_l;
						sum_r += pix_r * pix_r;
					}	/* for j */

				den = sqrt(sum_l * sum_r);

				if (den != 0)
					ncc = (sum) / den;
				else
					ncc = 0;

				if (ncc > *(max_array + width * y + left_x)) {
					*(disparity + width * y + left_x) = disp;	/*- min_disparity;*/
					*(max_array + width * y + left_x) = ncc;
				}	/* if */

				p_l++;
				p_r++;
				left_mean_ptr++;
				right_mean_ptr++;
			}	/* for left_x */
		}		/* for y */
	}			/* for disparity */

#ifdef STANDALONE
	free(left_means);
	free(right_means);
#endif

	printf("\n");
}				/* match_ZNCC_left */

#define	MATCH_RIGHT	match_ZNCC_right
#define MATCH_LEFT      match_ZNCC_left

#ifndef STANDALONE
#include	"glue.c"
#endif
