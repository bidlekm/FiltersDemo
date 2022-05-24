#include "Denoise.cuh"
#include <fstream>
#include <math.h>

__global__ void cudaGradientDescentDenoise(float* image, float* d, int width, int height, float step)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	/*compute gradient by column */
	if ((y < height) && (x < width) && (y > 0) && (x > 0))
	{
		d[y * width + x] =	((image[y * width + x] - image[y * width + x - 1]) / (sqrtf(powf(image[y * width + x] - image[y * width + x - 1], 2.0f) + powf(image[(y + 1) * width + x] - image[y * width + x], 2.0f)) + 0.001f))
							+ ((image[y * width + x] - image[(y - 1) * width + x]) / (sqrtf(powf(image[y * width + x + 1] - image[y * width + x], 2.0f) + powf(image[y * width + x] - image[(y - 1) * width + x], 2.0f)) + 0.001f))
		-((image[(y + 1) * width + x] + image[y * width + x + 1] - 2.0f * image[y * width + x]) / (sqrtf(powf(image[y * width + x + 1] - image[y * width + x], 2) + powf(image[(y + 1) * width + x] - image[y * width + x], 2) ) + 0.001f));

	}
	else
		d[y * width + x] = 0.0;

	__syncthreads();
	if(d[y * width + x] != 0.0)
	{
		d[y * width + x] = d[y * width + x] / abs(d[y * width + x]);
	}
	__syncthreads();
	if ((y < height) && (x < width) && (y > 0) && (x > 0))
		image[y * width + x] = image[y * width + x]- step* d[y * width + x];

}

int GradientDescentDenoise(int width, int height, unsigned char* data)
{
	int ImageSize = width * height;
	std::ofstream myfile;

	float* filteredImage = new float[ImageSize];
	for (int i = 0; i < ImageSize; ++i)
	{
		filteredImage[i] = data[i];
	}
	
	if (cudaSuccess != denoiseGradientDescent(filteredImage, width, height))
		return -1;

	for (int i = 0; i < ImageSize; ++i)
	{
		data[i] = clamp(filteredImage[i],0.0f,255.0f);
	}
	return 0;
}

cudaError_t denoiseGradientDescent(float* hostImage, int width, int height)
{
	cudaError_t cudaStatus;
	float* cudaImage;
	float* d;
	{
		CudaCall(cudaSetDevice(0));

		CudaCall(cudaMalloc((void**)&cudaImage, width * height * sizeof(float)));
		
		CudaCall(cudaMalloc((void**)&d, width * height * sizeof(float)));


	} //initCuda and variables

	// Copy input image from host memory to GPU buffers.
	CudaCall(cudaMemcpy(cudaImage, hostImage, width * height * sizeof(float), cudaMemcpyHostToDevice));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((int)ceil(width / 16), (int)ceil(height / 16));
	for (size_t i = 0; i < 50; i++)
	{
		cudaGradientDescentDenoise << <blocksPerGrid, threadsPerBlock >> > (cudaImage, d, width, height, 0.1f);
	}

	// Check for any errors launching the kernel
	CudaCall(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CudaCall(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	CudaCall(cudaMemcpy(hostImage, cudaImage, width*height* sizeof(float), cudaMemcpyDeviceToHost));

	CudaCall(cudaFree(cudaImage));
	CudaCall(cudaFree(d));

	return cudaSuccess;
}

