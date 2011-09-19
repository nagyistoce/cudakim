#include <cuda/cuda_utilities.h>

#include <common/error.h>

/**
 *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
 */
void CHECK_FOR_CUDA_ERROR_FUNCTION(const char* file, const char* line) {
	cudaError_t errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess) {
		exit_with_error("[file:%s line:%s] CUDA Error:", 
                        file, line, cudaGetErrorString(errorCode) );
	}
    errorCode = cudaThreadSynchronize();
    if (errorCode != cudaSuccess) {
		exit_with_error("[file:%s line:%s] CUDA sync Error:", 
                        file, line, cudaGetErrorString(errorCode) );
    }
}
