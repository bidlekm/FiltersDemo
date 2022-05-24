#include "Denoise.cuh"


__global__ void gradient(float* image, float* gradientRow, float* gradientColumn, int xsize, int ysize)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	/*compute gradient by row */
	if ((i < ysize - 1) && (j < xsize))
		gradientRow[i * xsize + j] = image[i * xsize + j] - image[(i + 1) * xsize + j];
	if ((i == ysize - 1) && (j < xsize))
		gradientRow[i * xsize + j] = 0;

	/*compute gradient by column */
	if ((i < ysize) && (j < xsize - 1))
		gradientColumn[i * xsize + j] = image[i * xsize + j] - image[i * xsize + j + 1];
	if ((i < ysize) && (j == xsize - 1))
		gradientColumn[i * xsize + j] = 0;

}

// --------------------------------------------------------------------------
// Compute conj row-wise and column-wise
// --------------------------------------------------------------------------

__global__ void conj(float* d_temp_dual_row, float* d_temp_dual_col, float* gradientRow, float* gradientColumn, int xsize, int ysize, float lambda)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	float sum;
	if ((i < ysize) && (j < xsize))
	{
		sum = (gradientColumn[i * xsize + j]) * (gradientColumn[i * xsize + j]) + (gradientRow[i * xsize + j]) * (gradientRow[i * xsize + j]);
		if ((sqrt(sum) / lambda - 1) > 0)
		{
			d_temp_dual_col[i * xsize + j] = gradientColumn[i * xsize + j] / (sqrt(sum) / lambda);
			d_temp_dual_row[i * xsize + j] = gradientRow[i * xsize + j] / (sqrt(sum) / lambda);
		}
		else
		{
			d_temp_dual_col[i * xsize + j] = gradientColumn[i * xsize + j];
			d_temp_dual_row[i * xsize + j] = gradientRow[i * xsize + j];
		}
	}

}

// --------------------------------------------------------------------------
// Compute adj 
// --------------------------------------------------------------------------

__global__ void  adjoint(float* d_adj, float* d_dual_row, float* d_dual_col, int xsize, int ysize)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < ysize - 1) && (j < xsize))
		d_adj[(i + 1) * xsize + j] = -(d_dual_row[i * xsize + j] - d_dual_row[(i + 1) * xsize + j]);
	if ((i == 0) && (j < xsize))
		d_adj[i * xsize + j] = -d_dual_row[i * xsize + j];
	if ((i < ysize) && (j < xsize - 1))
		d_adj[i * xsize + j + 1] = d_adj[i * xsize + j + 1] - (d_dual_col[i * xsize + j] - d_dual_col[i * xsize + j + 1]);
	if ((i < ysize) && (j == 0))
		d_adj[i * xsize + j] = d_adj[i * xsize + j] - d_dual_col[i * xsize + j];
}

__global__ void prox_tau_f(float* d_adj, float* d_sol, float* d_temp_sol, float* image, int xsize, int ysize, float tau)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < ysize) && (j < xsize))
	{
		d_sol[i * xsize + j] = ((d_temp_sol[i * xsize + j] - tau * d_adj[i * xsize + j]) + tau * image[i * xsize + j]) / (1 + tau);
	}
}

__global__ void diff(float* d_sol, float* d_temp_sol, float* d_diff, int xsize, int ysize)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < ysize) && (j < xsize))
	{
		d_diff[i * xsize + j] = 2 * d_sol[i * xsize + j] - d_temp_sol[i * xsize + j];
	}
}

__global__ void ascent(float* d_dual_row, float* d_dual_col, float* gradientRow, float* gradientColumn, int xsize, int ysize, float sigma)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < ysize) && (j < xsize))
	{
		gradientRow[i * xsize + j] = d_dual_row[i * xsize + j] + sigma * gradientRow[i * xsize + j];
		gradientColumn[i * xsize + j] = d_dual_col[i * xsize + j] + sigma * gradientColumn[i * xsize + j];
	}
}

__global__ void update_sol(float* d_temp_sol, float* d_sol, int xsize, int ysize, float rho)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < ysize) && (j < xsize))
	{
		d_temp_sol[i * xsize + j] = d_temp_sol[i * xsize + j] + rho * (d_sol[i * xsize + j] - d_temp_sol[i * xsize + j]);
	}
}

__global__ void upadate_dual(float* d_temp_dual_row, float* d_temp_dual_col, float* d_dual_row, float* d_dual_col, int xsize, int ysize, float rho)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if ((i < ysize) && (j < xsize))
	{
		d_temp_dual_row[i * xsize + j] = d_temp_dual_row[i * xsize + j] + rho * (d_dual_row[i * xsize + j] - d_temp_dual_row[i * xsize + j]);
		d_temp_dual_col[i * xsize + j] = d_temp_dual_col[i * xsize + j] + rho * (d_dual_col[i * xsize + j] - d_temp_dual_col[i * xsize + j]);
	}
}

cudaError_t denoiseDual(float* hostImage, int width, int height)
{
	float lambda = 1.3, tau = 0.05, sigma = 1 / tau / 8, rho = 1.9;
	float* cudaImage;
	float* gradientRow, * gradientColumn, * d_dual_col, * d_dual_row, * d_adj;
	float* d_diff, * d_temp_dual_row, * d_temp_dual_col, * d_temp_sol, * d_sol;
	{
		CudaCall(cudaSetDevice(0));

		CudaCall(cudaMalloc((void**)&cudaImage, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_sol, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_temp_sol, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&gradientRow, width * height * sizeof(float)));;
		CudaCall(cudaMalloc((void**)&gradientColumn, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_dual_row, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_dual_col, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_diff, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_temp_dual_row, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_temp_dual_col, width * height * sizeof(float)));
		CudaCall(cudaMalloc((void**)&d_adj, width * height * sizeof(float)));


	} //initCuda and variables

	// Copy input image from host memory to GPU buffers.
	CudaCall(cudaMemcpy(cudaImage, hostImage, width * height * sizeof(float), cudaMemcpyHostToDevice));
	// Launch a kernel on the GPU with one thread for each element.
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((int)ceil(width / 16), (int)ceil(height / 16));

	CudaCall(cudaMemcpy(hostImage, d_temp_sol, width * height * sizeof(float), cudaMemcpyDeviceToHost));
	/* Compute gradient row-wise and column-wise */
	gradient << <blocksPerGrid, threadsPerBlock >> > (cudaImage, gradientRow, gradientColumn, width, height);

	/* Initialize the dual solution */
	conj << <blocksPerGrid, threadsPerBlock >> > (d_temp_dual_row, d_temp_dual_col, gradientRow, gradientColumn, width, height, lambda);

	for (int i = 0; i < 100; i++)
	{
		adjoint << <blocksPerGrid, threadsPerBlock >> > (d_adj, d_temp_dual_row, d_temp_dual_col, width, height);

		/* compute current solution */
		prox_tau_f << <blocksPerGrid, threadsPerBlock >> > (d_adj, d_sol, d_temp_sol, cudaImage, width, height, tau);

		/* compute current dual solution */
		diff << <blocksPerGrid, threadsPerBlock >> > (d_sol, d_temp_sol, d_diff, width, height);
		gradient << <blocksPerGrid, threadsPerBlock >> > (d_diff, gradientRow, gradientColumn, width, height);
		ascent << <blocksPerGrid, threadsPerBlock >> > (d_temp_dual_row, d_temp_dual_col, gradientRow, gradientColumn, width, height, sigma);
		conj << <blocksPerGrid, threadsPerBlock >> > (d_dual_row, d_dual_col, gradientRow, gradientColumn, width, height, lambda);
		update_sol << <blocksPerGrid, threadsPerBlock >> > (d_temp_sol, d_sol, width, height, rho);
		upadate_dual << <blocksPerGrid, threadsPerBlock >> > (d_temp_dual_row, d_temp_dual_col, d_dual_row, d_dual_col, width, height, rho);
	}

	// Check for any errors launching the kernel
	CudaCall(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CudaCall(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	CudaCall(cudaMemcpy(hostImage, d_sol, width * height * sizeof(float), cudaMemcpyDeviceToHost));

	CudaCall(cudaFree(cudaImage));
	CudaCall(cudaFree(d_sol));
	CudaCall(cudaFree(d_temp_sol));
	CudaCall(cudaFree(gradientRow));
	CudaCall(cudaFree(gradientColumn));
	CudaCall(cudaFree(d_dual_row));
	CudaCall(cudaFree(d_dual_col));
	CudaCall(cudaFree(d_diff));
	CudaCall(cudaFree(d_temp_dual_row));
	CudaCall(cudaFree(d_temp_dual_col));
	CudaCall(cudaFree(d_adj));

	return cudaSuccess;
}

int PrimalDualDenoise(int width, int height, unsigned char* data)
{
	int ImageSize = width * height;
	std::ofstream myfile;

	float* filteredImage = new float[ImageSize];
	for (int i = 0; i < ImageSize; ++i)
	{
		filteredImage[i] = data[i];
	}

	if (cudaSuccess != denoiseDual(filteredImage, width, height))
		return -1;

	for (int i = 0; i < ImageSize; ++i)
	{
		data[i] = clamp(filteredImage[i], 0.0f, 255.0f);
	}
	return 0;
}