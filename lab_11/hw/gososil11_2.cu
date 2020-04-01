#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>

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

TIMER_T compute_time = 0;
TIMER_T device_time = 0;

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

int N;
int Nf;
#define BLOCK_SIZE 8
#define N_initial 1<<26
#define Nf_initial 256
#define N_COL 1<<13

int *X;
int *S_CPU;
int *S_GPU;

cudaError_t HW3_GPU(int *X, int *S_GPU);

__global__ void HW3_kernel(int *cuda_X, int *cuda_S_GPU,int *cuda_N, int *cuda_Nf)
{
	int row = blockDim.y*blockIdx.y + threadIdx.y;	
	int col = blockDim.x*blockIdx.x + threadIdx.x;
	int id = gridDim.x*blockDim.x*row + col;		
	
	int sum;
	sum = 0;

	for(int j=id-*cuda_Nf; j<=id+*cuda_Nf; j++){
		if(j<0 || j>=*cuda_N)
			continue;
		else
			sum += cuda_X[j];
	}
	cuda_S_GPU[id] = sum;
}

void HW3_CPU(int *X, int *S_CPU) {
	printf("***HW3_CPU Start!!\n");

	int sum;

	for(int i=0;i<N;i++){
		sum = 0;
		for(int j=i-Nf;j<=i+Nf;j++){
			if(j<0 || j>=N)
				continue;
			else
				sum += X[j];
		}
		S_CPU[i] = sum;
	}

	printf("***HW3_CPU End!!\n\n");
}

void make_bin_file(){
	printf("***Binary File make Start!!\n");
	FILE *fp = fopen("Cuda_HW3_input.bin", "wb");

	N = N_initial;
	Nf = Nf_initial;

	fwrite(&N, sizeof(int), 1, fp);
	fwrite(&Nf, sizeof(int), 1, fp);
	for(int i=0;i<N;i++){
		int num = (rand()%201) - 100;
		fwrite(&num, sizeof(int), 1, fp);
	}

	fclose(fp);
	printf("***Binary File make End!!\n\n");
}

void read_bin_file() {
	printf("***Binary File Read Start!!\n");
	FILE *fp = fopen("Cuda_HW3_input.bin", "rb");
	fread(&N, sizeof(int), 1, fp);
	fread(&Nf, sizeof(int), 1, fp);
	X = (int *)malloc(sizeof(int) * N);
	S_CPU = (int *)malloc(sizeof(int) * N);
	S_GPU = (int *)malloc(sizeof(int) * N);

	fread(X, sizeof(int), N, fp);

	fclose(fp);
	printf("***Binary File Read End!!\n\n");
}

void write_bin_file() {
	printf("***Binary File Write Start!!\n");
	FILE *fp = fopen("Cuda_HW3_output.bin", "wb");

	fwrite(&N, sizeof(int), 1, fp);
	fwrite(&Nf, sizeof(int), 1, fp);
	fwrite(S_CPU, sizeof(int), N, fp);

	fclose(fp);
	printf("***Binary File Write End!!\n\n");
}
int main()
{
	srand(time(NULL));
	make_bin_file();
	read_bin_file();
	CHECK_TIME_START;
	HW3_CPU(X, S_CPU);
	CHECK_TIME_END(compute_time);

	printf("***HW3_CPU : Time taken = %.6fms\n", compute_time);
	
	cudaError_t cudaStatus = HW3_GPU(X,S_GPU);
	if(cudaStatus != cudaSuccess){
		printf("HW3_GPU fail!\n");
		printf("end!!\n\n");
		return 0;
	}

	int i;
	for (i = 0; i < N; i++) {
		if (S_CPU[i] != S_GPU[i]) {
			printf("%d != %d\n", S_CPU[i], S_GPU[i]);
			break;
		}
	}
	if (i == N)
		printf("***Kernel execution Success!!\n\n");
		
	write_bin_file();
	printf("end!!\n\n");

	return 0;
}

cudaError_t HW3_GPU(int *X, int *S_GPU)
{
	printf("***HW3_GPU Start!!\n");
	cudaError_t cudaStatus;

	CUDA_CALL(cudaSetDevice(0))

	CHECK_TIME_INIT_GPU()
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int *cuda_X;
	int *cuda_S_GPU;
	int *cuda_N;
	int *cuda_Nf;

	CUDA_CALL(cudaMalloc(&cuda_X,(sizeof(int) * N)));
	CUDA_CALL(cudaMemcpy(cuda_X,X,(sizeof(int) * N),cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&cuda_N,(sizeof(int) * 1)));
	CUDA_CALL(cudaMemcpy(cuda_N,&N,(sizeof(int) * 1),cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&cuda_Nf,(sizeof(int) * 1)));
	CUDA_CALL(cudaMemcpy(cuda_Nf,&Nf,(sizeof(int) * 1),cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&cuda_S_GPU,(sizeof(int) * N)));

	dim3 dimBlock(BLOCK_SIZE*4, BLOCK_SIZE*2);

	int N_col = N_COL;
	dim3 dimGrid(N_col/dimBlock.x,(N/N_col)/dimBlock.y);

	CHECK_TIME_START_GPU()
	HW3_kernel <<< dimGrid, dimBlock >>> (cuda_X, cuda_S_GPU, cuda_N, cuda_Nf);
	CHECK_TIME_END_GPU(device_time)
	
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize())
	CUDA_CALL(cudaMemcpy(S_GPU,cuda_S_GPU,(sizeof(int) * N),cudaMemcpyDeviceToHost))
	
	printf("***HW3_GPU End!!\n\n");
	printf("***HW3_GPU : Time taken = %.6fms\n", device_time);
	
Error:
	CHECK_TIME_DEST_GPU();
	return cudaStatus;
}
