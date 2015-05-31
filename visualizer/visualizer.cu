/* 
	Written by Justinien Bouron.
	See visualizer.h for more details.
*/
#include <iostream>	/* For cout */
#include <stdio.h>
	#include <SFML/Graphics.hpp>
#include "visualizer.h"
#include "../generator/generator.h"

using namespace std;

#define UNITTEST_IMGW 640
#define UNITTEST_IMGH 640
#define UNITTEST_MAXITE 10000
#define UNITTEST_SHIFTX -0.743643887037151
#define UNITTEST_SHIFTY 0.13182590420533
#define UNITTEST_ZOOM 100000.0

int main(void){
	int* pixels = (int*)calloc(UNITTEST_IMGW*UNITTEST_IMGH, sizeof(int));
	if (pixels == NULL){
		cout << "Alloc error. Test failed." << endl;
		return -1;
	}

	PRECISION z = 1.0;

	sf::Image img;

	#define MAX_FILENAME 8192
	char name[MAX_FILENAME];
	
	while (z > 0){
		if (generate(pixels, UNITTEST_IMGW, UNITTEST_IMGH, UNITTEST_MAXITE, UNITTEST_SHIFTX, UNITTEST_SHIFTY, z) == NULL){
			 cout << "Generate failed." << endl;
			return -1;
		}		
	
		img.create(UNITTEST_IMGW, UNITTEST_IMGH, (sf::Uint8*)pixels);
		sprintf(name, "MSZ=%f.jpg", z);
		img.saveToFile(name);
		z *= 2 ;
		cout << "Zoom = " << z << endl;
	}
	
	return 0;
}
