#include <common/image2d.h>

#include <common/error.h>

#include <stdlib.h>

unsigned int image2d_get_number_of_elements(image2d* i) { 
    return i->width * i->height;
}

unsigned int image2d_get_size(image2d* i) {
    return image2d_get_number_of_elements(i) * i->channels; 
}

void image2d_init(image2d* i, unsigned int w, unsigned int h, unsigned int c) {
    i->width = w;
    i->height = h;
    i->channels = c;
    i->data = (BYTE*) malloc( image2d_get_size(i) * sizeof(BYTE) );
}

void image2d_free(image2d* i) {
    free( i->data );
}

void image2d_reverse_vertically(image2d* i) {
    BYTE* temp = i->data;
    i->data = (BYTE*) malloc( image2d_get_size(i) * sizeof(BYTE) );
    unsigned int x, y, c;
    for (x=0; x<i->width; x++)
        for (y=1; y<=i->height; y++)
            for (c=0; c<i->channels; c++)
                i->data[(x+(y-1)*i->width)*i->channels+c] = 
                    temp[(x+(i->height-y)*i->width)*i->channels+c];
    free(temp);
}

void image2d_load_from_raw(image2d* i, const char* file) {
    FILE * pFile;
    unsigned long lSize;
    size_t result;

    pFile = fopen( file , "rb" );
    if (pFile==NULL) {
        exit_with_error("error opening file: %s", file);
    }
    
    /* obtain file size */
    fseek(pFile , 0 , SEEK_END);
    lSize = ftell(pFile);
    rewind(pFile);

    if (lSize != image2d_get_size(i)) {
        exit_with_error("image and file size does not match, file: %s", file);
    }

    /* copy the file into the buffer */
    result = fread(i->data, 1, lSize, pFile);
    if (result != lSize) {
        exit_with_error("Reading error, file: %s", file); 
    }

    fclose(pFile);
  
    image2d_reverse_vertically(i);
}

void image2d_save_to_raw(image2d* image, const char* filename) {
    image2d_reverse_vertically(image);
    FILE* file = fopen(filename,"wb");
    if (!file) {
        exit_with_error("writing file failed, file: %s", file); 
    }
    fwrite(image->data, image2d_get_size(image), sizeof(BYTE), file);
    fflush(file);
    fclose(file);
}
