#include "Denoise.cuh"


__device__ float generate(curandState* globalState, int ind)
{
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void setup_kernel(curandState* state, unsigned long seed, int width)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int id = y * width + x;
	curand_init(seed, id, 0, &state[id]);
}

__global__ void cudaAddNoise(float* image, int width, int height, float* d_cost, curandState* globalState)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float cudaAddNoise;
	if ((i < height) && (j < width))
	{
		cudaAddNoise = image[i * width + j] + generate(globalState, i * width + j) * 20.0f;
		d_cost[i * width + j] = cudaAddNoise * cudaAddNoise;
		image[i * width + j] = (float)cudaAddNoise;
	}
}


cudaError_t addGaussianWhiteNoise(float* hostImage, int width, int height)
{
	cudaError_t cudaStatus;
	float* cudaImage;
	float* d_cost;
	{
		CudaCall(cudaSetDevice(0));

		CudaCall(cudaMalloc((void**)&cudaImage, width * height * sizeof(float)));

		CudaCall(cudaMalloc((void**)&d_cost, width * height * sizeof(float)));

	} //initCuda and variables

	// Copy input image from host memory to GPU buffers.
	CudaCall(cudaMemcpy(cudaImage, hostImage, width * height * sizeof(float), cudaMemcpyHostToDevice));


	curandState* devStates;
	cudaMalloc(&devStates, width * height * sizeof(curandState));
	srand(time(0));
	int seed = rand();
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((int)ceil(width / 16), (int)ceil(height / 16));
	setup_kernel << <blocksPerGrid, threadsPerBlock >> > (devStates, seed, width);
	cudaAddNoise << <blocksPerGrid, threadsPerBlock >> > (cudaImage, width, height, d_cost, devStates);
	CudaCall(cudaMemcpy(hostImage, cudaImage, width * height * sizeof(float), cudaMemcpyDeviceToHost));

	// Check for any errors launching the kernel
	CudaCall(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CudaCall(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	CudaCall(cudaMemcpy(hostImage, cudaImage, width * height * sizeof(float), cudaMemcpyDeviceToHost));

	CudaCall(cudaFree(cudaImage));
	CudaCall(cudaFree(d_cost));
}


int AddNoiseToImage(int width, int height, unsigned char* data)
{

	int ImageSize = width * height;
	std::ofstream myfile;

	float* noisyImage = new float[ImageSize];
	for (int i = 0; i < ImageSize; ++i)
	{
		noisyImage[i] = data[i];
	}


	addGaussianWhiteNoise(noisyImage, width, height);

	for (int i = 0; i < ImageSize; ++i)
	{
		data[i] = clamp(noisyImage[i], 0.0f, 255.0f);
	}
	return 0;
}