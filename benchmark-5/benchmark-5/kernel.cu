#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ctime"
#include <iostream>
#include <stdio.h>
#include <string>
#include <windows.h>
//-----------------------------------------------------------------------------------------------------------


#define MATRIX_SQUARE_SIZE		3
#define THREADS_PER_BLOCK		128
#define DISPLAY_MATRIX_ON		// If uncommented, matrices are displayed
//-----------------------------------------------------------------------------------------------------------


__global__ void multiplyKernel(const int* A, const int* B, int* C, int nRows);
__host__ cudaError_t multiplyKernerError(const int* A, const int* B, int* C, unsigned int size);
__host__ void display(int* M, int size, std::string title);
__host__ void initialize(int* M, int nRows);
//-----------------------------------------------------------------------------------------------------------


int main()
{
	srand(time(NULL));

	// Variables
	clock_t _bench;
	const int _n_rows = MATRIX_SQUARE_SIZE;
	const int _n_cells = MATRIX_SQUARE_SIZE * MATRIX_SQUARE_SIZE;
	size_t _size = _n_cells * sizeof(int);
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = 128; //(_n_cells + threadsPerBlock - 1) / threadsPerBlock;

	// Allocate input matrices in device memory
	int* _host_A = (int*)malloc(_size);
	int* _host_B = (int*)malloc(_size);
	int* _host_C = (int*)malloc(_size);

	// Initialize input matrices
	initialize(_host_A, _n_rows);
	initialize(_host_B, _n_rows);

	// Allocate three matrices (two inputs, one output) in device memory    .
	int* _device_A; cudaMalloc((void**)&_device_A, _size);
	int* _device_B; cudaMalloc((void**)&_device_B, _size);
	int* _device_C; cudaMalloc((void**)&_device_C, _size);

	// Copy input matrices from host memory to device memory (GPU buffers).
	cudaMemcpy(_device_A, _host_A, _size, cudaMemcpyHostToDevice);
	cudaMemcpy(_device_B, _host_B, _size, cudaMemcpyHostToDevice);

	// Invoke kernel on GPU with one thread for each cell in the result matrix
	_bench = clock();
	multiplyKernel << < blocksPerGrid, threadsPerBlock >> > (_device_A, _device_B, _device_C, _n_rows);
	_bench = clock() - _bench;

	// Copy the result from device memory to host memory
	cudaMemcpy(_host_C, _device_C, _size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(_device_A);
	cudaFree(_device_B);
	cudaFree(_device_C);

	std::cout << "\nElapsed time " << (float)_bench / CLOCKS_PER_SEC << " second(s)" << std::endl;

#ifdef DISPLAY_MATRIX_ON
	// Display the input matrices
	display(_host_A, _n_rows, "Matrix A");
	display(_host_B, _n_rows, "Matrix B");
	display(_host_C, _n_rows, "Matrix C");
#endif

	// Free host memory
	free(_host_A);
	free(_host_B);
	free(_host_C);

	system("pause");
	return 0;
}
//-----------------------------------------------------------------------------------------------------------


__global__ void multiplyKernel(const int* A, const int* B, int* C, int nRows)
{
	// Get the unique thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadID < nRows)
	{
		int i = threadID / nRows;
		int j = threadID % nRows;
		C[threadID] = 0;
		for (int k = 0; k<nRows; k++)
		{
			C[threadID] += A[(i*nRows) + k] * B[(k*nRows) + j];
		}
	}
}
//-----------------------------------------------------------------------------------------------------------


__host__ void display(int* M, int size, std::string title)
{
	std::cout << title << std::endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			std::cout << M[(i*size) + j] << " ";
		}
		std::cout << std::endl;
	}
}
//-----------------------------------------------------------------------------------------------------------


__host__ void initialize(int* M, int nRows)
{
	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nRows; j++)
		{
			M[(i*nRows) + j] = rand() % 10;
		}
	}
}
//-----------------------------------------------------------------------------------------------------------