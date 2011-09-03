#ifndef _BINARY_THRESHOLD_
#define _BINARY_THRESHOLD_

#include <image2d.h>

#ifdef __cplusplus
extern "C" {
#endif

void binary_threshold(image2d* image, BYTE threshold, BYTE low_val, BYTE high_val);

#ifdef __cplusplus
}
#endif

#endif /* _BINARY_THRESHOLD_ */
