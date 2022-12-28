#define _GNU_SOURCE /* See feature_test_macros(7) */

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

double setp1TimeArray[REPEAT];
double setp2TimeArray[REPEAT];
double setp3TimeArray[REPEAT];
double allTimeArray[REPEAT];
char setp1timeStr[100];
char setp2timeStr[100];
char setp3timeStr[100];
char alltimeStr[100];

ncclDataType_t test_types[ncclNumTypes] = {ncclInt8,
                                           ncclUint8,
                                           ncclInt32,
                                           ncclUint32,
                                           ncclInt64,
                                           ncclUint64,
                                           ncclHalf,
                                           ncclFloat,
                                           ncclDouble
#if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
                                           ,
                                           ncclBfloat16
#endif
};
const char* test_typenames[ncclNumTypes] = {"int8",
                                            "uint8",
                                            "int32",
                                            "uint32",
                                            "int64",
                                            "uint64",
                                            "half",
                                            "float",
                                            "double"
#if defined(__CUDA_BF16_TYPES_EXIST__) && NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
                                            ,
                                            "bfloat16"
#endif
};
int test_typenum = -1;

const char* test_opnames[] = {"sum", "prod", "max", "min", "avg", "mulsum"};
ncclRedOp_t test_ops[] = {
    ncclSum,
    ncclProd,
    ncclMax,
    ncclMin
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    ,
    ncclAvg
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
    ,
    ncclNumOps // stand in for ncclRedOpCreatePreMulSum() created on-demand
#endif
};

testResult_t bruckMain(void* sendbuff, void* recvbuff, void* tmpBuffer, size_t count, ncclDataType_t type,
                       ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, bool rotate, int repeat) {
	int i, pof2, block;
	int nRanks, rank;
	void* tmp_ptr;
	// timer tim;

	NCCLCHECK(ncclCommCount(comm, &nRanks));
	NCCLCHECK(ncclCommUserRank(comm, &rank));

	int* displs = (int*)malloc(sizeof(int) * nRanks);

#ifdef BRUCK_DEBUG
	// size_t rankOffset = count * wordSize(type);
	// size_t totalByteSize = count * nRanks * wordSize(type);
	// (WGT): Need to suppose nRank is a power of 2 ?
	int msbOfNRank = (int)msb(nRanks);
	int allBlockNeedSend = (nRanks / 2) * ccLog2(nRanks);
	int maxBlockPeerRoundSend = allBlockNeedSend / msbOfNRank;
	// int tmpBuffSize = count * maxBlockPeerRoundSend * wordSize(type);
	printf("[rank-[%d] Bruck info]:: msb:[%d], allBlockNeedSend:[%d], maxBlockPeerRoundSend:[%d], count[%d]\n", rank,
	       msbOfNRank, allBlockNeedSend, maxBlockPeerRoundSend, count);
#endif

	/* Do Phase 1 of the algorithim. Shift the data blocks on process i
	 * upwards by a distance of i blocks. Store the result in tmpBuffer. */
	/* (WGT): We cannot omit the first rotate, otherwise the result of all2all will be incorrect. */
	size_t rotateCount_1 = rank * count;
	size_t rotateCount_2 = (nRanks - rank) * count;
	CUDACHECK(cudaMemcpyAsync(((float*)tmpBuffer + rotateCount_2), sendbuff, rotateCount_1 * wordSize(type),
	                          cudaMemcpyDeviceToDevice, stream));
	CUDACHECK(cudaMemcpyAsync(tmpBuffer, ((float*)sendbuff + rotateCount_1), rotateCount_2 * wordSize(type),
	                          cudaMemcpyDeviceToDevice, stream));
	tmp_ptr = tmpBuffer;
	tmpBuffer = sendbuff;
	sendbuff = tmp_ptr;
	// cudaStreamSynchronize(stream);

	// setp1TimeArray[repeat] = tim.elapsed() * 1.0E6;
	// tim.reset();

#ifdef BRUCK_DEBUG
	dumpData<float>((float*)tmpBuffer, rank, count * nRanks, "tmpBuffer", true);
	dumpData<float>((float*)sendbuff, rank, count * nRanks, "sendbuff", true);
#endif

	pof2 = 1;
	while (pof2 < nRanks) {
		/* (WGT): Assuming nRanks is calculated from 0. */
		int nextRingPeer = (rank + pof2) % nRanks;
		int preRingPeer = (rank - pof2 + nRanks) % nRanks;

		/* Exchange all data blocks whose ith bit is 1 */
		/* (WGT): Copy send data to contiguous tmpBuffer (Device2Device) */
		int blockNeedSend = 0;
		for (block = 1; block < nRanks; block++) {
			if (block & pof2) {
				CUDACHECK(cudaMemcpyAsync(((float*)tmpBuffer + blockNeedSend * count),
				                          ((float*)sendbuff + block * count), count * wordSize(type),
				                          cudaMemcpyDeviceToDevice, stream));
				displs[blockNeedSend++] = block * count;
			}
		}
#ifdef BRUCK_DEBUG
		printf("[rank-[%d] Bruck round [%d]]:: next:[%d], pre:[%d], blockNeedSend:[%d], sendCount[%d]\n", rank, pof2,
		       nextRingPeer, preRingPeer, blockNeedSend, blockNeedSend * count);
		printf("[rank-[%d] Bruck round [%d]]:: displs[", rank, pof2);
		for (i = 0; i < blockNeedSend; i++)
			printf("%d,", displs[i]);
		printf("]\n");
#endif

		cudaStreamSynchronize(stream);

		/* (WGT): Do send/recv here */
		NCCLCHECK(ncclGroupStart());
		NCCLCHECK(ncclSend(tmpBuffer, blockNeedSend * count, type, nextRingPeer, comm, stream));
		NCCLCHECK(ncclRecv(recvbuff, blockNeedSend * count, type, preRingPeer, comm, stream));
		NCCLCHECK(ncclGroupEnd());

		cudaStreamSynchronize(stream);

#ifdef BRUCK_DEBUG
		dumpData<float>((float*)tmpBuffer, rank, blockNeedSend * count, "tmpBuffer", true);
		dumpData<float>((float*)recvbuff, rank, blockNeedSend * count, "recvbuff", true);
#endif

		/* (WGT): Copy recv data from recvbuff to sendbuff, use recv data to overwrite the send data.(Device2Device) */
		/* (WGT): Because the behavior of each Rank in each round of send/recv is symmetric, there is no need to send
		 * displs. */
		for (i = 0; i < blockNeedSend; i++)
			CUDACHECK(cudaMemcpyAsync(((float*)sendbuff + displs[i]), ((float*)recvbuff + i * count),
			                          count * wordSize(type), cudaMemcpyDeviceToDevice, stream));

		// cudaStreamSynchronize(stream);

		pof2 *= 2;
	}

	// setp2TimeArray[repeat] = tim.elapsed() * 1.0E6;
	// tim.reset();

	/* Rotate blocks in sendbuff upwards by (rank + 1) blocks. Need
	 * a temporary buffer of the same size as sendbuff. */
	if (rotate) {
		int rotateCount_1 = (rank + 1) * count;
		int rotateCount_2 = (nRanks - rank - 1) * count;
		CUDACHECK(cudaMemcpyAsync((float*)recvbuff + rotateCount_2, sendbuff, rotateCount_1 * wordSize(type),
		                          cudaMemcpyDeviceToDevice, stream));
		CUDACHECK(cudaMemcpyAsync(recvbuff, (float*)sendbuff + rotateCount_1, rotateCount_2 * wordSize(type),
		                          cudaMemcpyDeviceToDevice, stream));
	}

	/* Blocks are in the reverse order now (comm_size-1 to 0).
	 * Reorder them to (0 to comm_size-1) and store them in recvbuf. */
	if (rotate) {
		for (i = 0; i < nRanks; i++) {
			size_t recvOffset = (nRanks - i - 1) * count;
			size_t sendOffset = i * count;
			CUDACHECK(cudaMemcpyAsync(((float*)tmpBuffer + recvOffset), ((float*)recvbuff + sendOffset),
			                          count * wordSize(type), cudaMemcpyDeviceToDevice, stream));
		}

		// cudaStreamSynchronize(stream);
	}

	// setp3TimeArray[repeat] = tim.elapsed() * 1.0E6;

	// (WGT): Final recv data will be saved in tmpBuffer.
	return testSuccess;
}

testResult_t initData(BRUCK_TYPE** sendbuff, BRUCK_TYPE** recvbuff, BRUCK_TYPE** tmpBuffer, BRUCK_TYPE** syncSendBuff,
                      BRUCK_TYPE** syncRecvBuff, int myRank, int size, int count, bool reset) {
	int j;
	BRUCK_TYPE* hostBuffer = (BRUCK_TYPE*)malloc(size * sizeof(BRUCK_TYPE));
	if (!reset) {
		CUDACHECK(cudaMalloc((void**)sendbuff, size * sizeof(BRUCK_TYPE)));
		CUDACHECK(cudaMalloc((void**)recvbuff, size * sizeof(BRUCK_TYPE)));
		CUDACHECK(cudaMalloc((void**)tmpBuffer, size * sizeof(BRUCK_TYPE)));
		CUDACHECK(cudaMalloc((void**)syncSendBuff, 128 * sizeof(BRUCK_TYPE)));
		CUDACHECK(cudaMalloc((void**)syncRecvBuff, 128 * sizeof(BRUCK_TYPE)));
		CUDACHECK(cudaMemset((void*)*tmpBuffer, 0, size * sizeof(BRUCK_TYPE)));
	}

	CUDACHECK(cudaMemset((void*)*sendbuff, 0, size * sizeof(BRUCK_TYPE)));
	CUDACHECK(cudaMemset((void*)*recvbuff, 0, size * sizeof(BRUCK_TYPE)));

	for (j = 0; j < size; j++)
		hostBuffer[j] = (BRUCK_TYPE)(myRank * 100 + j / count);

	CUDACHECK(cudaMemcpy((void*)*sendbuff, (void*)hostBuffer, size * sizeof(BRUCK_TYPE), cudaMemcpyHostToDevice));
	free(hostBuffer);
	return testSuccess;
}

bool cmpAll2AllResult(BRUCK_TYPE* buff1, BRUCK_TYPE* buff2, int nRanks, int count, int rank) {
	int i, j, re1 = 0, re2 = 0;
	int size = nRanks * count * sizeof(BRUCK_TYPE);
	BRUCK_TYPE* buff1_h = (BRUCK_TYPE*)malloc(size);
	BRUCK_TYPE* buff2_h = (BRUCK_TYPE*)malloc(size);
	int* expected = (int*)malloc(sizeof(int) * nRanks);
	CUDACHECK(cudaMemcpy(buff1_h, buff1, size, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaMemcpy(buff2_h, buff2, size, cudaMemcpyDeviceToHost));

	for (i = 0; i < nRanks; i++) {
		expected[i] = 0;
		for (j = 0; j < nRanks; j++) {
			expected[i] += (j * 100 + i) * count;
		}
	}

	for (i = 0; i < nRanks * count; i++) {
		re1 += (int)buff1_h[i];
		re2 += (int)buff2_h[i];
	}

	free(buff1_h);
	free(buff2_h);
#ifdef BRUCK_DEBUG
	printf("expected:%d, re1:%d, re2: %d\n", expected[rank], re1, re2);
#endif
	return (expected[rank] == re1) && (expected[rank] == re2);
}

testResult_t ncclAlltoAll(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op, int root,
                          ncclComm_t comm, cudaStream_t stream) {
	int nRanks;
	NCCLCHECK(ncclCommCount(comm, &nRanks));
	size_t rankOffset = count * wordSize(type);

	NCCLCHECK(ncclGroupStart());
	for (int r = 0; r < nRanks; r++) {
		NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
		NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
	}
	NCCLCHECK(ncclGroupEnd());
	return testSuccess;
}

int all2AllBruck(int rank, int nRanks, int localRank, size_t msgSize) {
	// NCCL and CUDA stuff
	ncclUniqueId id;
	ncclComm_t comm;
	cudaStream_t s;
	BRUCK_TYPE *sendbuff, *recvbuff, *tmpBuffer, *syncSendBuff, *syncRecvBuff;
	size_t all2allSize = msgSize * nRanks;
	size_t count = msgSize / sizeof(BRUCK_TYPE);

	// cudaSetDevice must call before cudaStreamCreate
	CUDACHECK(cudaSetDevice(localRank));
	CUDACHECK(cudaStreamCreate(&s));
	// Init data.
	initData(&sendbuff, &recvbuff, &tmpBuffer, &syncSendBuff, &syncRecvBuff, rank, all2allSize, count, false);
	NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, rank));

	for (int j = 0; j < REPEAT; j++) {
		bruckMain(sendbuff, recvbuff, tmpBuffer, count, ncclFloat, (ncclRedOp_t)0, 0, comm, s, false, j);
		cudaStreamSynchronize(s);
		initData(&sendbuff, &recvbuff, &tmpBuffer, &syncSendBuff, &syncRecvBuff, rank, all2allSize, count, true);
	}

	ncclAlltoAll(sendbuff, recvbuff, count, ncclFloat, (ncclRedOp_t)0, 0, comm, s);

	if (cmpAll2AllResult(tmpBuffer, recvbuff, nRanks, count, rank))
		printf("Bruck all2all result is correct!\n");
	else
		printf("Bruck all2all result is not correct!\n");

	return 0;
}
