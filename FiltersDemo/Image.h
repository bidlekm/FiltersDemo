#pragma once
#include <fstream>
#include <iostream>
#include <algorithm>
#include "Image.h"
#include "Denoise.cuh"

class Image {
	int width, height;
	unsigned char* data;

public:
	Image(const char* path);
	unsigned char* GetData();
	~Image();
	int GetWidth();
	int GetHeight();
};