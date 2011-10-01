/*
 * deviceProp.cu
 *
 *  Created on: 26/09/2011
 *      Author: kimbjerge
 */

// includes, system
#include <stdio.h>

// includes, project
#include <cutil_inline.h>

////////////////////////////////////////////////////////////////////////////////
// Printing properties of NVIDIA GPU card
////////////////////////////////////////////////////////////////////////////////
void PrintDeviceProperties(void)
{
	int devID;
    cudaDeviceProp deviceProps;

	devID = cutGetMaxGflopsDeviceId();
    cudaSetDevice( devID );
	cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));

	printf("Device %d: [%s]\n", devID, deviceProps.name);
	printf("  Major revision number:            %d\n", deviceProps.major);
	printf("  Minor revision number:            %d\n", deviceProps.minor);
	printf("  Total amount of global memory:    %d\n", deviceProps.totalGlobalMem);
	printf("  Number of multiprocessors (SM):   %d\n", deviceProps.multiProcessorCount);
	printf("  Max. threads per multiprocessor   768\n"); // Gforce 9400 + G80
	printf("  Max. blocks per multiprocessor    8\n");   // Gforce 9400 + G80
	printf("  Execute multiple kernels:         %s\n", (deviceProps.concurrentKernels == 0 ? "no" : "yes"));
	printf("  Constant memory:                  %d\n", deviceProps.totalConstMem);
	printf("  Shared memory per block:          %d\n", deviceProps.sharedMemPerBlock);
	printf("  Registers per block:              %d\n", deviceProps.regsPerBlock);
	printf("  Warp size:                        %d\n", deviceProps.warpSize);
	printf("  Max. threads per block:           %d\n", deviceProps.maxThreadsPerBlock);
	printf("  Max. dimension of block:          [%d,%d,%d]\n", deviceProps.maxThreadsDim[0], deviceProps.maxThreadsDim[1], deviceProps.maxThreadsDim[2]);
	printf("  Max. dimension of grid:           [%d,%d,%d]\n", deviceProps.maxGridSize[0], deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);
	printf("  Max. memory pitch:                %d\n", deviceProps.memPitch);
	printf("  Texture alignment:                %d\n", deviceProps.textureAlignment);
	printf("  Clock rate:                       %d Hz\n", deviceProps.clockRate);
	printf("  Concurrent copy and exe:          %s\n", (deviceProps.deviceOverlap == 0 ? "no" : "yes"));
	printf("\n");

}




