#ifndef _IMAGE_2D_
#define _IMAGE_2D_

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char BYTE;

typedef struct {
    unsigned int width, height;
    unsigned int channels; /* number of channels per pixel */
    BYTE* data;
} image2d;

unsigned int image2d_get_number_of_elements(image2d* i);
unsigned int image2d_get_size(image2d* i);
void image2d_init(image2d* i, unsigned int w, unsigned int h, unsigned int c);
void image2d_free(image2d* i);
void image2d_reverse_vertically(image2d* i);
void image2d_load_from_raw(image2d* i, const char* file);
void image2d_save_to_raw(image2d* i, const char* file);

#ifdef __cplusplus
}
#endif

#endif /* _IMAGE_2D_ */
