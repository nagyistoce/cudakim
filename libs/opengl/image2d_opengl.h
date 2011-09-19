#ifndef _IMAGE_2D_OPENGL_
#define _IMAGE_2D_OPENGL_

#include <common/image2d.h>

#ifdef __cplusplus
extern "C" {
#endif

unsigned int image2d_upload_to_gpu(image2d* i);
void image2d_remove_from_gpu_memory(unsigned int textureid);

#ifdef __cplusplus
}
#endif

#endif /* _IMAGE_2D_OPENGL_ */
