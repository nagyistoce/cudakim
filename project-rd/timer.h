/*
 * timer.c
 *
 *  Created on: 07/09/2011
 *      Author: kimbjerge
 */

#include <cutil_inline.h>

////////////////////////////////////////////////////////////////////////////////
// Timer functions
////////////////////////////////////////////////////////////////////////////////
void inline CreateTimer(unsigned int *timer)
{
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

void inline DeleteTimer(unsigned int timer)
{
	cutilCheckError(cutDeleteTimer(timer));
}

