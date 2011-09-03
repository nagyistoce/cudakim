#include <thresholding.h>

#include <stdio.h>
#include <stdlib.h>

#include <cuda_utilities.h>

__global__
void binary_threshold_kernel(BYTE* in_out_image, unsigned int numElements, BYTE threshold, BYTE low_val, BYTE high_val) {
    //calculate a unique index into image (device) memory for each kernel invocation 
    const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < numElements) { // make sure we are inside the image domain
        BYTE pixel_value = in_out_image[idx];
        if (pixel_value > threshold)
            in_out_image[idx] = high_val;
        else
            in_out_image[idx] = low_val;
    }
}

void binary_threshold(image2d* image, BYTE threshold, BYTE low_val, BYTE high_val ) {
    unsigned int numElements = image2d_get_number_of_elements(image);

    // copy data to device
    BYTE* data_on_device;
    cudaMalloc( (void**) &data_on_device, numElements*sizeof(BYTE));
    cudaMemcpy(	data_on_device, image->data, numElements*sizeof(BYTE), cudaMemcpyHostToDevice );
    CHECK_FOR_CUDA_ERROR();

    // setup dimensions of grid/blocks.
    dim3 blockDim(512,1,1);
    dim3 gridDim((unsigned int) ceil((double)(numElements/blockDim.x)), 1, 1 );

    // invoke kernel
    binary_threshold_kernel<<< gridDim, blockDim >>>( data_on_device, numElements, threshold, low_val, high_val );
    CHECK_FOR_CUDA_ERROR();

    // copy data to host
    cudaMemcpy(	image->data, data_on_device, numElements*sizeof(BYTE), cudaMemcpyDeviceToHost );
    CHECK_FOR_CUDA_ERROR();
}
