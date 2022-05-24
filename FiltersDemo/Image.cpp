#include "Image.h"

Image::Image(const char* path)
{
	const char* imagepath = path;
	// Data read from the header of the BMP file
	unsigned char header[54]; // Each BMP file begins by a 54-bytes header
	unsigned int dataPos;     // Position in the file where the actual data begins
	unsigned int imageSize;   // = width*height*3

	// Open the file
	FILE* file = fopen(imagepath, "rb");
	if (!file)
	{
		printf("Image could not be opened\n");
	}

	if (fread(header, 1, 54, file) != 54)
	{ // If not 54 bytes read : problem
		printf("Not a correct BMP file\n");
	}
	if (header[0] != 'B' || header[1] != 'M')
	{
		printf("Not a correct BMP file\n");
	}

	std::cout << "Image loaded: " << path <<std::endl << std::endl;


	dataPos = *(int*)&(header[0x0A]);
	imageSize = *(int*)&(header[0x22]);
	std::cout << "Size: " << imageSize << std::endl << std::endl;
	width = *(int*)&(header[0x12]);
	std::cout << "Width: " << width << std::endl << std::endl;
	height = *(int*)&(header[0x16]);
	std::cout << "Height: " << height << std::endl << std::endl;

	// Some BMP files are misformatted, guess missing information
	if (imageSize == 0)    imageSize = width * height * 3; // 3 : one byte for each Red, Green and Blue component
	if (dataPos == 0)      dataPos = 54; // The BMP header is done that way

	// Create a buffer
	data = new unsigned char[imageSize];

	// Read the actual data from the file into the buffer
	fread(data, 1, imageSize, file);
	unsigned char* r = new unsigned char[imageSize / 3];
	unsigned char* g = new unsigned char[imageSize / 3];
	unsigned char* b = new unsigned char[imageSize / 3];
	int n = 0;
	for (int i = 0; i < imageSize / 3; ++i)
	{
		r[i] = data[n++];
		g[i] = data[n++];
		b[i] = data[n++];
	}
	delete data;
	data = new unsigned char[imageSize / 3];
	for (int i = 0; i < imageSize / 3; ++i)
	{
		unsigned char lum;
		lum = (r[i] * 0.33) + (g[i] * 0.33) + (b[i] * 0.33);
		data[i] = lum;
	}
	delete r;
	delete g;
	delete b;

	fclose(file);
}

unsigned char* Image::GetData()
{
	return data;
}

int Image::GetWidth()
{
	return width;
}

int Image::GetHeight()
{
	return height;
}

Image::~Image()
{
	delete data;
}