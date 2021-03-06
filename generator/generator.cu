/*
	Written by JBouron. See generator.h for more details.
*/
#include <iostream>	/* For cerr */

#include "generator.h"

#define  BITS_PRECISION 512

using namespace std;

__device__ int colorize(int i, int max_ite){
	int lvl = ((float)i/max_ite)*255;
	return lvl << 24 | lvl << 16 | lvl << 8 | lvl;
}

__global__ void compute_fractal(int* pixels, PRECISION shift_x, PRECISION shift_y, int img_w, int img_h, PRECISION zoom, int max_ite){
	/* Position x and y of the point to be tested in the image. */
	int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

	/* Index of the pixel in the pixels array */
        int px_idx = y * img_w + x;

	/* We test if we are out of bounds */
        if (px_idx > img_w * img_h - 1){
                return;
        }else{
		/* Simple algorithm which test if the given point is in the Mandelbrot Set or not. */	
                PRECISION c_r = shift_x + (x - img_w/2)/zoom;
                PRECISION c_i = shift_y + (y - img_h/2)/zoom;
                PRECISION z_r = 0;
                PRECISION z_i = 0;

                int i = 0;
                do{
                        PRECISION tmp = z_r;
                        z_r = z_r*z_r - z_i*z_i + c_r;
                        z_i = 2*z_i*tmp + c_i;
                        i ++;
                }while (z_r*z_r + z_i*z_i < 4 && i < max_ite);
		
		/* We compute the color of this point. */
                pixels[px_idx] = colorize(i, max_ite);
        }	
}

int* generate(int* pixels, int img_w, int img_h, int max_ite, PRECISION shift_x, PRECISION shift_y, PRECISION zoom_level){
	/* We first test the validity of the arguments. */
	if (img_w > 0 && img_h > 0 && max_ite > 0 && zoom_level > 0 && pixels != NULL){
		/* Allocating space for the pixels. */
		size_t alloc_size = img_w*img_h*sizeof(int);
	
		/* Allocating memory for the pixels, but this time on the device. */
		int* device_pixels;
		if (cudaMalloc((void**) &device_pixels, alloc_size) == cudaErrorMemoryAllocation){
			cerr << "An error has occured while allocating on device memory." << endl;
			return NULL;
		}

		/* Defining the number of threads per block and the number of blocks. */
		dim3 threadsPerBlock(16, 16);
	        dim3 numBlocks(img_w / threadsPerBlock.x, img_h / threadsPerBlock.y);	

		/* Calling the kernels */
		compute_fractal<<<numBlocks, threadsPerBlock>>>(device_pixels, shift_x, shift_y, img_w, img_h, zoom_level, max_ite);

		/* Waiting for the end of the computation. */
		cudaDeviceSynchronize();

		/* Copying the pixels. */
		cudaMemcpy(pixels, device_pixels, alloc_size, cudaMemcpyDeviceToHost);
		cudaFree(device_pixels);

		return pixels;
	}
	else return NULL;
}
