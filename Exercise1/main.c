#include <stdlib.h>

#include <thresholding.h>
#include <error.h>
#include <image2d.h>

int main(int argc, char** argv) {
    if (argc != 6)
        exit_with_error("invalid parameters, use: <input file> <width> <height> <output file> <threshold value>\n");

    const char* input_file = argv[1];
    const unsigned int width = atoi(argv[2]);
    const unsigned int height = atoi(argv[3]);
    const char* output_file = argv[4];
    const unsigned int threshold = atoi(argv[5]);

    image2d* image = (image2d*)malloc(sizeof(image2d));
    image2d_init(image, width, height, 1);
    image2d_load_from_raw(image, input_file);

    /*
     * do a thresholding of the image, pixel of less
     * than 'threshold' are replaced by '0' and '255' otherwise
     */
    binary_threshold(image, threshold, 0, 255);

    image2d_save_to_raw(image, output_file);

    image2d_free(image);
    return EXIT_SUCCESS;
}
