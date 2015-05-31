/*
	Created by Justinien Bouron.
	
	This file is responsible to create a fractal with the given properties.

*/

#ifndef GENERATOR_H_INCLUDED
#define GENERATOR_H_INCLUDED

#include <SFML/Graphics.hpp>

typedef double PRECISION; /* Precision of the calculation, the macro allows to change it easily. */

/* Note : for this application, all images will be 32 bits per pixel images. */

/* The kernel for the GPUs 
	img : pointer on the array which will  receive the data.
	shift_x : x position of the point centered on the image.
        shift_y : y position of the point centered on the image.
	img_w : witdh of the image.
	img_h : height of the image.
	zoom : zoom of the image.
	max_ite : maximum number of iteration for this pixel.
	color : the function used to colorize the image.
*/
__global__ void compute_fractal(int* pixels, PRECISION shift_x, PRECISION shift_y, int img_w, int img_h, PRECISION zoom, int max_ite);

/* The generator itself, return a pointer on a SDL_Surface on success, NULL otherwise. 
	pixels : array which will hold the result.
	img_w : width of the image to be created.
	img_h : height of the image to be created.
	max_ite : threshold for the number of iteration per pixel.
	shift_x : x position of the point centered on the image.
	shift_y : y position of the point centered on the image.
	zoom_level : zoom of the image to be created.
	color : the function used to colorize the image.
*/
int* generate(int* pixels, int img_w, int img_h, int max_ite, PRECISION shift_x , PRECISION shift_y, PRECISION zoom_level);

#endif
