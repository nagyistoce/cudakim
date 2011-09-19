#ifndef _CUDA_UTILITIES_
#define _CUDA_UTILITIES_

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  Should never be used in the code, use CHECK_FOR_CUDA_ERROR(); instead
 */
void CHECK_FOR_CUDA_ERROR_FUNCTION(const char* file, const char* line);

#ifdef __cplusplus
}
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/**
 *  Checks for CUDA errors and report if an error was detected.
 *  inspired by cutil.h: CUT_CHECK_ERROR
 *  NOTE: This call must not be used inside CUDA kernels
 */
/* #if _DEBUG */
#define CHECK_FOR_CUDA_ERROR() CHECK_FOR_CUDA_ERROR_FUNCTION(__FILE__,TOSTRING(__LINE__))
/* #else */
/* #define CHECK_FOR_CUDA_ERROR() */
/* #endif */

#endif /* _CUDA_UTILITIES_ */
