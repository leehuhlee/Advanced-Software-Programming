#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>


#define BLOCK_SIZE 4, 32
#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;
#define USE_CPU_TIMER 1
#define USE_GPU_TIMER 1
#if USE_CPU_TIMER == 1
__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }
#else
#define CHECK_TIME_START
#define CHECK_TIME_END(a)
#endif

#if USE_GPU_TIMER == 1
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

void create_device_timer()
{
	CUDA_CALL(cudaEventCreate(&cuda_timer_start));
	CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_device_timer()
{
	CUDA_CALL(cudaEventDestroy(cuda_timer_start));
	CUDA_CALL(cudaEventDestroy(cuda_timer_stop));
}

inline void start_device_timer()
{
	cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
	cudaEventSynchronize(cuda_timer_stop);

	cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }
#else
#define CHECK_TIME_INIT_GPU()
#define CHECK_TIME_START_GPU()
#define CHECK_TIME_END_GPU(a)
#define CHECK_TIME_DEST_GPU()
#endif

TIMER_T compute_time = 0;
TIMER_T device_time = 0;

int N;

int *h_Fibonacci_number;
unsigned *h_fibonacci_value;
unsigned *h_fibonacci_value_GPU;

cudaError_t Fibonacci_GPU(int *h_Fibo_n, unsigned *h_Fibo_v, unsigned *h_Fibo_v_GPU);

__global__ void Fibonacci_kernel(int *h_Fibo_n, unsigned *h_Fibo_v_GPU)
{
	int v = 0;
	float sqrt_5, x_0, x_1, tmp_0, tmp_1;
	int row = blockDim.y*blockIdx.y + threadIdx.y;
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int id = gridDim.x*blockDim.x*row + col;

	sqrt_5 = sqrtf(5.0f);
	x_0 = (1.0f + sqrt_5) / 2.0f;
	x_1 = (1.0f - sqrt_5) / 2.0f;
	tmp_0 = tmp_1 = 1.0f;

	for (v = 0; v < h_Fibo_n[id]; v++) {
			tmp_0 *= x_0; 
			tmp_1 *= x_1;
		}
	h_Fibo_v_GPU[id] = (unsigned)((tmp_0 - tmp_1) / sqrt_5 + 0.5f);

}

void Fibonacci_CPU(int *h_Fibo_n, unsigned *h_Fibo_v) {
	printf("***Fibonacci_CPU Start!!\n");
	int i, j;

	float sqrt_5, x_0, x_1, tmp_0, tmp_1;

	sqrt_5 = sqrtf(5.0f);
	x_0 = (1.0f + sqrt_5) / 2.0f; x_1 = (1.0f - sqrt_5) / 2.0f;

	for (i = 0; i < N; i++) {
		tmp_0 = tmp_1 = 1.0f;
		for (j = 0; j < h_Fibo_n[i]; j++) {
			tmp_0 *= x_0; tmp_1 *= x_1;
		}
		h_Fibo_v[i] = (unsigned)((tmp_0 - tmp_1) / sqrt_5 + 0.5f);
	}
	printf("***Fibonacci_CPU End!!\n\n");
}

void read_bin_file() {
	printf("***Binary File Read Start!!\n");
	FILE *fp = fopen("Fibonacci_number.bin", "rb");
	fread(&N, sizeof(int), 1, fp);

	h_Fibonacci_number = (int *)malloc(sizeof(int) * N);
	h_fibonacci_value = (unsigned *)malloc(sizeof(unsigned) * N);
	h_fibonacci_value_GPU = (unsigned *)malloc(sizeof(unsigned) * N);

	fread(h_Fibonacci_number, sizeof(int), N, fp);

	fclose(fp);
	printf("***Binary File Read End!!\n\n");
}

int main()
{
	float timeavg = 0.0f;
	read_bin_file();
	CHECK_TIME_START;
	Fibonacci_CPU(h_Fibonacci_number, h_fibonacci_value);
	CHECK_TIME_END(compute_time);

	cudaError_t cudaStatus;

	printf("***Fibonacci_GPU Start!!\n");
	for(int i = 0; i < 10; i++) {
		cudaStatus = Fibonacci_GPU(h_Fibonacci_number, h_fibonacci_value, h_fibonacci_value_GPU);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "combine_two_arrays_GPU failed!");
			return 1;
		}
		timeavg += device_time;
	}
	printf("***Fibonacci_GPU End!!\n\n");

	int i;
	for (i = 0; i < N; i++) {
		if (h_fibonacci_value[i] != h_fibonacci_value_GPU[i]) {
			printf("index : %d\n", i);
			printf("%u != %u\n", h_fibonacci_value[i], h_fibonacci_value_GPU[i]);

		}
	}
	if (i == N)
		printf("***Kernel execution Success!!\n\n");
	printf("***CPU Time taken = %.6fms\n", compute_time);
	printf("***GPU Time taken = %.6fms\n", timeavg/10);
	printf("end!!\n\n");

	return 0;
}

cudaError_t Fibonacci_GPU(int *h_Fibo_n, unsigned *h_Fibo_v, unsigned *h_Fibo_v_GPU)
{
	CHECK_TIME_INIT_GPU()
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0); 
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	
	int *d_h_Fibo_n;
	unsigned *d_h_Fibo_v_GPU;
	size_t size = N * sizeof(int);

	CUDA_CALL(cudaMalloc(&d_h_Fibo_n, size)) 
	CUDA_CALL(cudaMemcpy(d_h_Fibo_n, h_Fibo_n, size, cudaMemcpyHostToDevice)) 

	size = N * sizeof(unsigned);
	CUDA_CALL(cudaMalloc(&d_h_Fibo_v_GPU, size)) 
	
	dim3 dimBlock(4, 32);
	dim3 dimGrid((N/8192)/dimBlock.x, (N/8192)/dimBlock.y);

	CHECK_TIME_START_GPU()
		Fibonacci_kernel << < dimGrid, dimBlock >> > (d_h_Fibo_n, d_h_Fibo_v_GPU);
	CHECK_TIME_END_GPU(device_time)
	
	CUDA_CALL(cudaDeviceSynchronize()) 
	CUDA_CALL(cudaMemcpy(h_Fibo_v_GPU, d_h_Fibo_v_GPU, size, cudaMemcpyDeviceToHost))



Error:
	cudaFree(d_h_Fibo_n);
	cudaFree(d_h_Fibo_v_GPU);

	CHECK_TIME_DEST_GPU()
	return cudaStatus;
}
