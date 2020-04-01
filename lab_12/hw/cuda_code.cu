#pragma once

#include "cuda_code.cuh"

#include <stdio.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <Windows.h>
#include <time.h>
#include <assert.h>

#if USE_CPU_TIMER == 1
__int64 start, freq, end;
#define CHECK_TIME_START { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }
#else
#define CHECK_TIME_START
#define CHECK_TIME_END(a)
#define BLOCK_SIZE (1 << 6)													// CUDA 커널 thread block 사이즈

#define BLOCK_WIDTH (1 << 3)
#define BLOCK_HEIGHT (BLOCK_SIZE / BLOCK_WIDTH)
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

__constant__ float constant_gaussian_kernel[ 25 ];
extern __shared__ int sharedBuffer[ ];

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	Gaussian 필터링을 하는 커널
//	shared memory를 사용하지 않는다
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Gaussian_kernel_no_shared(IN unsigned char *d_bitmaps, OUT unsigned char *d_Gaussian, long width, long height) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int id = width*row + col;
	int sumX = 0;

	if(row<=0 || row >= height || col<=0 || col >= width){
		d_Gaussian[id]=0;
	}
	else{
		for(int i=-2; i<3; i++){
			for(int j=-2; j<3; j++){
				int pixel = d_bitmaps[(row + i)* width + (col + j)];
				sumX += pixel * constant_gaussian_kernel[(i+2)*5 + j+2];
			}

			__syncthreads();

			int ans = abs(sumX) / 273;
			if(ans > 255) ans = 255;
			if(ans<0) ans=0;
			d_Gaussian[row*width + col] = ans;
		}
	}
		
	//	return;

	// 왜 id = width * row + col 인지 line 81과 함께 생각해 볼것
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	Gaussian 필터링을 하는 커널
//	shared memory를 사용한다.
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void Gaussian_kernel_shared(INOUT unsigned char *d_bitmaps, OUT unsigned char *d_Gaussian, long width, long height) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int id = width*row + col;


	if(row >= height || col >= width) return;

	if(thread_id == 0)
		for(int i = 0; i < 36; i++)
			for(int j = 0; j < 36; j++)
				sharedBuffer[i*36+j] = d_bitmaps[id+(i-2)*width+(j-2)];

	__syncthreads();

	unsigned char tmp = 0;
	for(int i = 0; i < 5; i++){
		for(int j = 0; j < 5; j++){
			if(0 <= row+i-2 && row+i-2 < height && 0 <= col+j-2 && col+j-2 < width)
				tmp += sharedBuffer[(i+threadIdx.y)*36+j+threadIdx.x] * constant_gaussian_kernel[i*5+j];
		}
	}
	
	d_Gaussian[id] = tmp;
	// 왜 id = width * row + col 인지 line 98과 함께 생각해 볼것
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	Constant variable 인 gaussian kernel을 설정하는 함수
//	후에 gaussian filtering 에서 사용한다.
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Set_Gaussain_Kernel(){
	float _1 = 1.0f / 256.0f;
	float _4 = _1 * 4;
	float _6 = _1 * 6;
	float _16 = _1 * 16;
	float _24 = _1 * 24;
	float _36 = _1 * 36;

	float *p_gaussian_kernel = new float[25];

	p_gaussian_kernel[0] = p_gaussian_kernel[4] = p_gaussian_kernel[20] = p_gaussian_kernel[24] = _1;
	p_gaussian_kernel[1] = p_gaussian_kernel[3] = p_gaussian_kernel[5] = p_gaussian_kernel[9]= _4;
	p_gaussian_kernel[15] = p_gaussian_kernel[19] = p_gaussian_kernel[21] = p_gaussian_kernel[23] = _4;
	p_gaussian_kernel[2] = p_gaussian_kernel[10] = p_gaussian_kernel[14] = p_gaussian_kernel[22] = _6;
	p_gaussian_kernel[6] = p_gaussian_kernel[8] = p_gaussian_kernel[16] = p_gaussian_kernel[18] = _16;
	p_gaussian_kernel[7] = p_gaussian_kernel[11] =p_gaussian_kernel[13] = p_gaussian_kernel[17] = _24;
	p_gaussian_kernel[12] = _36;

	cudaMemcpyToSymbol( constant_gaussian_kernel, p_gaussian_kernel, sizeof( float ) * 25 );

	delete[] p_gaussian_kernel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//	
//	커널을 실행하기 전 필요한 자료들 준비 및 커널을 실행할 디바이스를 설정
//	Shared_flag 입력 시 NO_SHARED 나 SHARED 중 한 개의 매크로를 넣으면
//	flag값에 맞는 커널을 실행
//	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float Do_Gaussian_on_GPU(IN unsigned char *p_bitmaps, OUT unsigned char *p_Gaussian, long width, long height, int Shared_flag)
{
	Set_Gaussain_Kernel();
	CUDA_CALL(cudaSetDevice(0));

	// block size 결정
	// width + blockDim.x - 1 과 height + blockDim.y - 1 을 왜 했는지 이해할 것
	dim3 blockDim(32, 32);																						
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);	

	switch (Shared_flag)
	{
	case NO_SHARED:
		//Todo
		Gaussian_kernel_no_shared << <gridDim, blockDim >> > (d_Bitmaps, d_Gaussian, width, height);
		break;
	case SHARED:
		//Todo
		Gaussian_kernel_shared  << <gridDim, blockDim , sizeof(unsigned char)*36*36 >> > (d_Bitmaps, d_Gaussian, width, height);
		break;
	}

	return device_time;
}