#define _GNU_SOURCE /* See feature_test_macros(7) */

#include "bruck.h"
#include "client.h"
#include "common.h"
#include "config.h"
#include "debug.h"
#include "ib.h"
#include "nccl.h"
#include "numa.h"
#include "server.h"
#include "setup_ib.h"
#include "sock.h"
#include <dlfcn.h>
#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ulimit.h>

#define BRUCK_TYPE float
#define START_COUNT 30

struct All2AllInfo {

	int nDevs;
	int nRanks;
	int base;
	int count_per_block;
	int count_per_rank;
	size_t total_buff_size;
	size_t buff_size;
	size_t block_size;

	// vars who needs adding index to vist.
	long* max_times;
	long* min_times;
	long* avg_times;
	// long* tts;
	int* max_idxs;
	int* min_idxs;
	int** range;

	BRUCK_TYPE** host_sendbuff;
	BRUCK_TYPE** host_recvbuff;
	BRUCK_TYPE** dev_sendbuff;
	BRUCK_TYPE** dev_recvbuff;

	cudaStream_t* streams;
	ncclComm_t* comms;
};

testResult_t initData_n(struct All2AllInfo* info) {

	for (int i = 0; i < info->nDevs; i++) {
		int dev_rank = info->base + i;
		char* rank_i_addr = (char*)info->host_sendbuff + i * info->buff_size;
		for (int j = 0; j < info->nRanks; j++) {
			char* block_j_rank_i_addr = rank_i_addr + j * info->block_size;
			memset(block_j_rank_i_addr, dev_rank * 100 + j, info->block_size);
		}
	}

	for (int i = 0; i < info->nDevs; i++) {
		CUDACHECK(cudaMemcpy(info->dev_sendbuff + i, info->host_sendbuff + i, info->buff_size, cudaMemcpyHostToDevice));
		memset(info->host_sendbuff + i, 0, info->buff_size);
	}

	return testSuccess;
}

testResult_t allocHostBuff(struct All2AllInfo* info) {
	info->host_sendbuff = (BRUCK_TYPE**)malloc(info->nDevs * sizeof(BRUCK_TYPE*));
	info->host_recvbuff = (BRUCK_TYPE**)malloc(info->nDevs * sizeof(BRUCK_TYPE*));

	CUDACHECK(cudaHostAlloc((void**)info->host_sendbuff, info->total_buff_size, cudaHostAllocDefault));
	CUDACHECK(cudaHostAlloc((void**)info->host_recvbuff, info->total_buff_size, cudaHostAllocDefault));

	memset((void*)info->host_sendbuff, 0, info->total_buff_size);
	memset((void*)info->host_recvbuff, 0, info->total_buff_size);

	return testSuccess;
}

testResult_t allocCudaBuff(struct All2AllInfo* info) {
	info->dev_sendbuff = (BRUCK_TYPE**)malloc(info->nDevs * sizeof(BRUCK_TYPE*));
	info->dev_recvbuff = (BRUCK_TYPE**)malloc(info->nDevs * sizeof(BRUCK_TYPE*));

	for (int i = 0; i < info->nDevs; i++) {
		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaStreamCreate(info->streams + i));
		CUDACHECK(cudaMalloc((void**)info->dev_sendbuff + i, info->buff_size));
		CUDACHECK(cudaMalloc((void**)info->dev_recvbuff + i, info->buff_size));
		// CUDACHECK(cudaMemset(info->dev_sendbuff[i], 0, info->buff_size));
		CUDACHECK(cudaMemset(info->dev_recvbuff[i], 0, info->buff_size));
	}

	return testSuccess;
}

testResult_t initBuff(struct All2AllInfo* info) {
	allocHostBuff(info);
	allocCudaBuff(info);
	return testSuccess;
}

void print_result(struct All2AllInfo* info) {
	// log("Rank-[%d], block_size:[%ld] MB, all2allSize:[%ld] MB, nic_flow_size:[%3.lf] MB avg_time:[%3.lf]", rank,
	//     info->block_size / (1024 * 1024), all2allSize / (1024 * 1024), nic_flow_size / (1024 * 1024),
	//     (double)(diffInNanos) / (1e3 * REPEAT));

	log("Base-[%d], nDevs-[%d], nRanks-[%d]", info->base, info->nDevs, info->nRanks);
	log("block_size[%3.lf] MB, count_per_block[%d], count_per_rank[%d], all2allSize[%ld] MB",
	    (double)(info->block_size / (double)(1024 * 1024)), info->count_per_block, info->count_per_rank,
	    info->buff_size / (1024 * 1024));
}

int all2AllBruck_nGPUs(int nRanks, int nDevs, size_t block_size, int base, int rank) {
	// NCCL and CUDA stuff
	struct All2AllInfo* info = (struct All2AllInfo*)calloc_numa(sizeof(struct All2AllInfo));
	int n, i;
	ncclUniqueId id;

	info->nDevs = nDevs;
	info->nRanks = nRanks;
	info->base = base;
	info->block_size = block_size;
	info->count_per_block = info->block_size / sizeof(BRUCK_TYPE);
	info->count_per_rank = info->count_per_block * nRanks;
	info->buff_size = info->block_size * nRanks;
	info->total_buff_size = info->buff_size * nDevs;

	info->streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDevs);
	info->comms = (ncclComm_t*)calloc_numa(sizeof(ncclComm_t) * nDevs);
	info->max_times = (long*)calloc_numa(sizeof(long) * nDevs);
	info->min_times = (long*)calloc_numa(sizeof(long) * nDevs);
	info->avg_times = (long*)calloc_numa(sizeof(long) * nDevs);
	// info->tts = (long*)calloc_numa(sizeof(long) * nDevs);
	info->max_idxs = (int*)calloc_numa(sizeof(int) * nDevs);
	info->min_idxs = (int*)calloc_numa(sizeof(int) * nDevs);
	info->range = (int**)calloc_numa(sizeof(int*) * nDevs);
	for (int i = 0; i < nDevs; i++)
		info->range[i] = (int*)calloc_numa(sizeof(int) * 10);

	log("Base-[%d], nDevs-[%d], nRanks-[%d]", info->base, info->nDevs, info->nRanks);
	log("block_size[%3.lf]MB, count_per_block[%d], count_per_rank[%d], all2allSize[%ld]MB",
	    (double)(info->block_size / (double)(1024 * 1024)), info->count_per_block, info->count_per_rank,
	    info->buff_size / (1024 * 1024));

	// generating NCCL unique ID at one process and broadcasting it to all
	if (info->base == 0) {
		ncclGetUniqueId(&id);
		for (i = 1; i < nRanks; i++) {
			n = sock_write(peer_sockfd[i], (void*)&id, sizeof(ncclUniqueId));
			CHECK(n == sizeof(ncclUniqueId), "ncclUniqueId send error.");
		}
	} else {
		n = sock_read(peer_sockfd[0], (void*)&id, sizeof(ncclUniqueId));
		CHECK(n == sizeof(ncclUniqueId), "ncclUniqueId recv error.");
	}
	log("ncclUniqueId broadcast success!");

	// cudaSetDevice must call before cudaStreamCreate
	CHECK(initBuff(info) == testSuccess, "buff init");
	CHECK(initData_n(info) == testSuccess, "data init");

	NCCLCHECK(ncclGroupStart());
	for (int i = 0; i < nDevs; i++) {
		CUDACHECK(cudaSetDevice(i));
		NCCLCHECK(ncclCommInitRank(info->comms + i, info->nRanks, id, info->base + i));
	}
	NCCLCHECK(ncclGroupEnd());

	// REPEAT
	long tt;
	struct timespec time1, time2;
	for (int i = 0; i < REPEAT; i++) {
		sock_barrier(nRanks, rank);

		clock_gettime(CLOCK_MONOTONIC, &time1);

		for (int j = 0; j < info->nDevs; j++)
			ncclAlltoAll(info->dev_sendbuff[j], info->dev_recvbuff[j], info->count_per_rank, ncclFloat, info->comms[j],
			             info->streams[j]);

		for (int j = 0; j < info->nDevs; j++)
			cudaStreamSynchronize(info->streams[j]);

		clock_gettime(CLOCK_MONOTONIC, &time2);
		tt = (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);

		if (tt > info->max_times[0]) {
			info->max_times[0] = tt;
			info->max_idxs[0] = i;
		}

		if (tt < info->min_times[0]) {
			info->min_times[0] = tt;
			info->min_idxs[0] = i;
		}

		info->avg_times[0] += tt;
		// log("Rank-[%d], round-[%d] time [%.3lf] us", rank, i, (double)tt / 1000);
		info->range[0][(int)(tt / (long)1e6) / 10]++;

		initData_n(info);
	}

	log("max_time:[%.3lf] idx-[%d], min_time:[%3.lf] idx-[%d], avg_time:[%3.lf]", (double)info->max_times[0] / 1e3,
	    info->max_idxs[0], (double)info->min_times[0] / 1e3, info->min_idxs[0],
	    (double)info->avg_times[0] / (1e3 * REPEAT));

error:

	for (int i = 0; i < info->nDevs; i++) {
		ncclCommDestroy(info->comms[i]);
	}
	return 0;
}
