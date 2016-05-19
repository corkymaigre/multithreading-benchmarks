#pragma region INCLUDES
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ctime"
#include <conio.h>			// using _getch()
#include <ctime>
#include <iomanip>			// using setfill() and setw()
#include <iostream>			// using cout, cin, and cerr
#include <stdio.h>
#include <stdlib.h>
#include <string>			// using string
#include <time.h>
namespace win32
{
#include <Windows.h>
}
#pragma endregion
//-----------------------------------------------------------------------------------------------------------

#pragma region DEFINES
#define MATRIX_SQUARE_SIZE		2000
#define THREADS_PER_BLOCK		128
#define BLOCS_PER_GRID			128
//#define DISPLAY_MATRIX_ON		// If uncommented, matrices are displayed
#define BENCHMARK_ON			// if uncommented, benchmark is applying

#define ERR_100					"cudaMalloc failed for _device_A"
#define ERR_101					"cudaMalloc failed for _device_B"
#define ERR_102					"cudaMalloc failed for _device_C"
#define ERR_200					"cudaMemcpy failed for copy host to device of A"
#define ERR_201					"cudaMemcpy failed for copy host to device of B"
#define ERR_202					"cudaMemcpy failed for copy host to device of C"
#define ERR_203					"cudaMemcpy failed for copy device to host of A"
#define ERR_204					"cudaMemcpy failed for copy device to host of B"
#define ERR_205					"cudaMemcpy failed for copy device to host of C"

#pragma endregion
//-----------------------------------------------------------------------------------------------------------


#pragma region CLASSES
class timer
{
	win32::LARGE_INTEGER start_time_;
public:
	timer()
	{
		QueryPerformanceCounter(&start_time_);
	}
	void restart()
	{
		QueryPerformanceCounter(&start_time_);
	}
	double elapsed() const
	{
		win32::LARGE_INTEGER end_time, frequency;
		QueryPerformanceCounter(&end_time);
		QueryPerformanceFrequency(&frequency);
		return (double(end_time.QuadPart - start_time_.QuadPart)) * 1000 / frequency.QuadPart;
	}
};
#pragma endregion
//-----------------------------------------------------------------------------------------------------------

#pragma region FUNCTION DECLARATIONS
__global__ void multiplyKernel(const int* A, const int* B, int* C, int nRows);
//__host__ cudaError_t multiplyKernerError(const int* A, const int* B, int* C, unsigned int size);
__host__ void display(int* M, int size, std::string title);
__host__ void initialize(int* M, int nRows);
__host__ void get_error(cudaError_t cudaStatus, std::string errorCode);
#pragma endregion
//-----------------------------------------------------------------------------------------------------------


#ifdef BENCHMARK_ON				// benchmark for speed tests
clock_t _clock;					// benchmark with clock_t
timer _timer;					// benchmark with Timer class
double _elapsed_time;			// elapsed time
#endif // BENCHMARK_ON
//-----------------------------------------------------------------------------------------------------------


#pragma region MAIN
int main()
{
	srand(time(NULL));

	clock_t _bench;
	const int _n_rows = MATRIX_SQUARE_SIZE;
	const int _n_cells = MATRIX_SQUARE_SIZE * MATRIX_SQUARE_SIZE;
	size_t _size = _n_cells * sizeof(int);
	int threadsPerBlock = THREADS_PER_BLOCK;
	int blocksPerGrid = BLOCS_PER_GRID; //(_n_cells + threadsPerBlock - 1) / threadsPerBlock;

	// Allocate input matrices in device memory
	int* _host_A = (int*)malloc(_size);
	int* _host_B = (int*)malloc(_size);
	int* _host_C = (int*)malloc(_size);

	// Initialize input matrices
	initialize(_host_A, _n_rows);
	initialize(_host_B, _n_rows);

	// Allocate three matrices (two inputs, one output) in device memory    .
	int* _device_A; get_error(cudaMalloc((void**)&_device_A, _size), ERR_100);
	int* _device_B; get_error(cudaMalloc((void**)&_device_B, _size), ERR_101);
	int* _device_C; get_error(cudaMalloc((void**)&_device_C, _size), ERR_102);

	// Copy input matrices from host memory to device memory (GPU buffers).
	get_error(cudaMemcpy(_device_A, _host_A, _size, cudaMemcpyHostToDevice), ERR_200);
	get_error(cudaMemcpy(_device_B, _host_B, _size, cudaMemcpyHostToDevice), ERR_201);

	// Invoke kernel on GPU with one thread for each cell in the result matrix
#ifdef BENCHMARK_ON
	_clock = clock();								// start the clock
	_timer.restart();								// start the timer
	multiplyKernel << < blocksPerGrid, threadsPerBlock >> > (_device_A, _device_B, _device_C, _n_rows);
	cudaDeviceSynchronize();
	_elapsed_time = _timer.elapsed();		// stop the timer
	_clock = clock() - _clock;			// stop the clock
#else
	Multiply(data);
#endif // BENCHMARK_ON


	// Copy the result from device memory to host memory
	get_error(cudaMemcpy(_host_C, _device_C, _size, cudaMemcpyDeviceToHost), ERR_205);

	// Free device memory
	cudaFree(_device_A);
	cudaFree(_device_B);
	cudaFree(_device_C);


#ifdef DISPLAY_MATRIX_ON
	// Display the input matrices
	display(_host_A, _n_rows, "Matrix A");
	display(_host_B, _n_rows, "Matrix B");
	display(_host_C, _n_rows, "Matrix C");
#endif

#ifdef BENCHMARK_ON
	std::cout << "Elapsed time with clock :\n\n";
	std::cout << "\t" << (float)_clock / CLOCKS_PER_SEC << " second(s)\n";
	std::cout << "\t" << (float)_clock << " milisecond(s)\n";
	std::cout << "\nElapsed time with timer :\n\n";
	std::cout << "\t" << _elapsed_time / 1000 << " second(s)\n";
	std::cout << "\t" << _elapsed_time << " milisecond(s)\n\n\n\n";
#endif // BENCHMARK_ON



	// Free host memory
	free(_host_A);
	free(_host_B);
	free(_host_C);



	system("pause");
	return 0;
}
#pragma endregion
//-----------------------------------------------------------------------------------------------------------


__global__ void multiplyKernel(const int* A, const int* B, int* C, int nRows)
{
	// Get the unique thread ID
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	while (threadID < nRows*nRows)
	{
		int i = threadID / nRows;
		int j = threadID % nRows;
		C[threadID] = 0;
		for (int k = 0; k<nRows; k++)
		{
			C[threadID] += A[(i*nRows) + k] * B[(k*nRows) + j];
		}
		threadID += THREADS_PER_BLOCK * BLOCS_PER_GRID;
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


__host__ void get_error(cudaError_t cudaStatus, std::string errorCode)
{
	if (cudaStatus != cudaSuccess)
	{
		std::cout << errorCode;
	}
}
//-----------------------------------------------------------------------------------------------------------

