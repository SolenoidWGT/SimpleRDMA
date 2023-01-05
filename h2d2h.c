#include "cuda_runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CUDACHECK(cmd)                                                                            \
	do {                                                                                          \
		cudaError_t e = cmd;                                                                      \
		if (e != cudaSuccess) {                                                                   \
			printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(-1);                                                                             \
		}                                                                                         \
	} while (0)

int main() {
	// each process is using two GPUs
	int nDev = 1;
	int size = 8 * 1024 * 1024; // 5344 * 1536
	int localRank = 0;
	struct timespec now, end;

	float** sendbuff_cuda = (float**)malloc(nDev * sizeof(float*));
	float** sendbuff_host = (float**)malloc(nDev * sizeof(float*));
	// cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);

	// picking GPUs based on localRank
	for (int i = 0; i < nDev; ++i) {
		CUDACHECK(cudaSetDevice(localRank * nDev + i));
		CUDACHECK(cudaMalloc((void**)sendbuff_cuda + i, size * sizeof(float)));
		CUDACHECK(cudaMemset((void*)sendbuff_cuda[i], 1, size * sizeof(float)));
		// CUDACHECK(cudaStreamCreate(s+i));
	}

	for (int i = 0; i < nDev; ++i) {
		CUDACHECK(cudaMallocHost((void**)&sendbuff_host[i], size * sizeof(float)));
		memset(sendbuff_host[i], 0, size * sizeof(float));
	}

	CUDACHECK(
	    cudaMemcpy((void*)sendbuff_cuda[0], (void*)sendbuff_host[0], size * sizeof(float), cudaMemcpyHostToDevice));
	CUDACHECK(
	    cudaMemcpy((void*)sendbuff_host[0], (void*)sendbuff_cuda[0], size * sizeof(float), cudaMemcpyDeviceToHost));
	// H2D
	clock_gettime(CLOCK_MONOTONIC, &now);
	for (int i = 0; i < 100; i++) {
		CUDACHECK(
		    cudaMemcpy((void*)sendbuff_cuda[0], (void*)sendbuff_host[0], size * sizeof(float), cudaMemcpyHostToDevice));
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	long long usetime = (end.tv_sec * 1000000000 + end.tv_nsec) - (now.tv_sec * 1000000000 + now.tv_nsec);
	printf("H2D-[%ld]B time use %.4f ms\n", size * sizeof(float), 1.0 * usetime / (1.0 * 1000 * 1000 * 100));

	// D2H
	clock_gettime(CLOCK_MONOTONIC, &now);
	for (int i = 0; i < 100; i++) {
		CUDACHECK(cudaMemcpy(sendbuff_host[0], sendbuff_cuda[0], size * sizeof(float), cudaMemcpyDeviceToHost));
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	long long usetime2 = (end.tv_sec * 1000000000 + end.tv_nsec) - (now.tv_sec * 1000000000 + now.tv_nsec);
	printf("D2H-[%ld]B time use %.4f ms\n", size * sizeof(float), 1.0 * usetime2 / (1.0 * 1000 * 1000 * 100));
	printf("all time %.4f ms\n", (1.0 * usetime2 + 1.0 * usetime) / (1.0 * 1000 * 1000 * 100));

	CUDACHECK(cudaFree(sendbuff_cuda[0]));
}