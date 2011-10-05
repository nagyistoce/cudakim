/*
 * timer.c
 *
 * Functions used to compute computation time
 * calculates the average time on 8 measurements
 *
 *  Created on: 07/09/2011
 *      Author: kimbjerge
 */

#include <cutil_inline.h>

#define WINDOW 8
static float timerAverage[WINDOW];
static unsigned int idx;

////////////////////////////////////////////////////////////////////////////////
// Timer functions
////////////////////////////////////////////////////////////////////////////////
void inline CreateTimer(unsigned int *timer)
{
	idx = 0;
	for (int i = 0; i < WINDOW; i++)
		timerAverage[i] = 0;
	cutilCheckError(cutCreateTimer(timer));
    cutilCheckError(cutResetTimer(*timer));
}

void inline StartTimer(unsigned int timer)
{
    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStartTimer(timer));
}

void inline StopTimer(unsigned int timer)
{
    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStopTimer(timer));
    if (idx < WINDOW)
    	timerAverage[idx++] = cutGetAverageTimerValue(timer);
}

void inline RestartTimer(unsigned int timer)
{
    cutilCheckError(cutResetTimer(timer));
    cutilSafeCall(cudaThreadSynchronize());
    cutilCheckError(cutStartTimer(timer));
}

float inline GetTimer(unsigned int timer)
{
	return cutGetAverageTimerValue(timer);
}

float inline GetAverage(unsigned int timer)
{
	if (idx == WINDOW)
	{
		float sum = 0;
		for (int i = 0; i < WINDOW; i++)
			sum += timerAverage[i];
		sum /= WINDOW;
		idx = 0;
		return sum;
	}
	return 0;
}

void inline DeleteTimer(unsigned int timer)
{
	cutilCheckError(cutDeleteTimer(timer));
}

