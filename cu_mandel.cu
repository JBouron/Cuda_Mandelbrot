#include <stdio.h>
#include <math.h>

/* some defines : */
#define IMG_SIZE_W 16384
#define IMG_SIZE_H 16384
#define IMG_ZOOM 10000.0
#define PRECISION float

#define SHIFT_X −0.1528
#define SHIFT_Y 1.0397

typedef unsigned char px;

__global__ void compute_fractal(px* img, int img_w, int img_h, PRECISION zoom, int max_ite){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int img_idx = y * img_w + x;
	img[img_idx] = 32;
	if (img_idx > img_w * img_h - 1){
		return;
	}else{
		PRECISION c_r = (x + 0.1528 - img_w/2)/zoom;
		PRECISION c_i = (y - 1.0397 - img_h/2)/zoom;
		PRECISION z_r = 0;	
		PRECISION z_i = 0;
		PRECISION i = 0;
		do{
			PRECISION tmp = z_r;
			z_r = z_r*z_r - z_i*z_i + c_r;
			z_i = 2*z_i*tmp + c_i;
			i ++;
		}while (z_r*z_r + z_i*z_i < 4 && i < max_ite);
		if (i == max_ite) img[img_idx] = 254;
		else img[img_idx] = 0;
		return;
	}
}

int main(int argc, char* argv[]){
	px* img = (px*)calloc(IMG_SIZE_W*IMG_SIZE_H, sizeof(px));
	px* d_img;
	cudaMalloc(&d_img, IMG_SIZE_W*IMG_SIZE_H*sizeof(px));
//	cudaMemcpy(d_img, img, IMG_SIZE_W*IMG_SIZE_H*sizeof(px), cudaMemcpyHostToDevice);	
	dim3 threadsPerBlock(16, 16);
    	dim3 numBlocks(IMG_SIZE_W / threadsPerBlock.x, IMG_SIZE_H / threadsPerBlock.y);
	clock_t beg = clock();
	compute_fractal<<<numBlocks, threadsPerBlock>>>(d_img, IMG_SIZE_W, IMG_SIZE_H, IMG_ZOOM, 10000);

	cudaDeviceSynchronize();
	clock_t end = clock();
	cudaMemcpy(img, d_img, IMG_SIZE_W*IMG_SIZE_H*sizeof(px), cudaMemcpyDeviceToHost);

	cudaFree(d_img);

	printf("Total computation time = %f\n", ((float)(end-beg))/CLOCKS_PER_SEC);
	free(img);
	return 0;
}
