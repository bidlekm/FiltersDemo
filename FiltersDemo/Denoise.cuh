#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>

#define CudaCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

template<typename T>
inline T clamp(T val, T min, T max)
{
	T temp = val < min ? min : val;
	return temp > max ? max : temp;
}


//Gradient Descent Denoise
int GradientDescentDenoise(int width, int height, unsigned char* data);
cudaError_t denoiseGradientDescent(float* hostImage, int width, int height);
__global__ void cudaGradientDescentDenoise(float* image, float* d, int width, int height, float step);


//Gaussian White Noise
int AddNoiseToImage(int width, int height, unsigned char* data);
cudaError_t addGaussianWhiteNoise(float* hostImage, int width, int height);
__device__ float generate(curandState* globalState, int ind);
__global__ void setup_kernel(curandState* state, unsigned long seed, int xsize);
__global__ void cudaAddNoise(float* image, int width, int height, float* d_cost, curandState* globalState);

//Primal Dual Denoise
int PrimalDualDenoise(int width, int height, unsigned char* data);
cudaError_t denoiseDual(float* hostImage, int width, int height);
__global__ void gradient(float* image, float* gradientRow, float* gradientColumn, int xsize, int ysize);
__global__ void conj(float* d_temp_dual_row, float* d_temp_dual_col, float* gradientRow, float* gradientColumn, int xsize, int ysize, float lambda);
__global__ void adjoint(float* d_adj, float* d_dual_row, float* d_dual_col, int xsize, int ysize);
__global__ void prox_tau_f(float* d_adj, float* d_sol, float* d_temp_sol, float* d_noisyGray, int xsize, int ysize, float tau);
__global__ void diff(float* d_sol, float* d_temp_sol, float* d_diff, int xsize, int ysize);
__global__ void ascent(float* d_dual_row, float* d_dual_col, float* gradientRow, float* gradientColumn, int xsize, int ysize, float sigma);
__global__ void update_sol(float* d_temp_sol, float* d_sol, int xsize, int ysize, float rho);
__global__ void upadate_dual(float* d_temp_dual_row, float* d_temp_dual_col, float* d_dual_row, float* d_dual_col, int xsize, int ysize, float rho);