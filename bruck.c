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

double setp1TimeArray[REPEAT];
double setp2TimeArray[REPEAT];
double setp3TimeArray[REPEAT];
double allTimeArray[REPEAT];
char setp1timeStr[100];
char setp2timeStr[100];
char setp3timeStr[100];
char alltimeStr[100];

enum All2AllType { BRUCK_NCCL, ONE2ONE_NCCL, INTRAINTER_NCCL, INTRAINTER_SYNC2_NCCL };

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
// For libnccl's < 2.13
char const* ncclGetLastError(ncclComm_t comm) { return ""; }

static size_t wordSize(ncclDataType_t type) {
	switch (type) {
	case ncclChar:
#if NCCL_MAJOR >= 2
	// case ncclInt8:
	case ncclUint8:
#endif
		return 1;
	case ncclHalf:
#if defined(__CUDA_BF16_TYPES_EXIST__)
	case ncclBfloat16:
#endif
		// case ncclFloat16:
		return 2;
	case ncclInt:
	case ncclFloat:
#if NCCL_MAJOR >= 2
	// case ncclInt32:
	case ncclUint32:
		// case ncclFloat32:
#endif
		return 4;
	case ncclInt64:
	case ncclUint64:
	case ncclDouble:
		// case ncclFloat64:
		return 8;
	default:
		return 0;
	}
}

testResult_t aligned_cuda_malloc(size_t bytes, size_t alignedment, void** buff) {
	if (alignedment == 0 || alignedment == 1)
		CUDACHECK(cudaMalloc(buff, bytes));
	else if ((alignedment & (alignedment - 1)) != 0) {
		log("alignedment is not correct!");
		return testInternalError;
	}
	CUDACHECK(cudaMalloc(buff, bytes + alignedment));
	*buff = (void*)(((size_t)buff + alignedment) & ~((size_t)alignedment - 1));
	// ((void**)(*buff))[-1] = *buff;
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

// int H2D2H(void* cpu_buff, void* gpu_buff, size_t size) {}

testResult_t bruckCPU(void* sendbuff, void* recvbuff, void* tmpBuffer, size_t count, ncclDataType_t type,
                      ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, bool rotate, int repeat) {
	int i, pof2, block;
	int nRanks, rank;
	void* tmp_ptr;
	// timer tim;

	NCCLCHECK(ncclCommCount(comm, &nRanks));
	NCCLCHECK(ncclCommUserRank(comm, &rank));

	int* displs = (int*)malloc(sizeof(int) * nRanks);

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

		cudaStreamSynchronize(stream);

		/* (WGT): Do send/recv here */
		// NCCLCHECK(ncclGroupStart());
		// NCCLCHECK(ncclSend(tmpBuffer, blockNeedSend * count, type, nextRingPeer, comm, stream));
		// NCCLCHECK(ncclRecv(recvbuff, blockNeedSend * count, type, preRingPeer, comm, stream));
		// NCCLCHECK(ncclGroupEnd());
		IBSendRecvP2P(nRanks, rank, tmpBuffer, recvbuff, nextRingPeer, preRingPeer,
		              blockNeedSend * count * wordSize(type));

		// cudaStreamSynchronize(stream);

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
testResult_t bruckNCCL(void* sendbuff, void* recvbuff, void* tmpBuffer, size_t count, ncclDataType_t type,
                       ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, bool rotate, int repeat,
                       cudaStream_t s_send, cudaStream_t s_recv) {
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
	/* (WGT): Do send/recv here */
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

		// cudaStreamSynchronize(s_send);

		// cudaStreamSynchronize(stream);
		NCCLCHECK(ncclGroupStart());
		NCCLCHECK(ncclSend(tmpBuffer, blockNeedSend * count, type, nextRingPeer, comm, stream));
		NCCLCHECK(ncclRecv(recvbuff, blockNeedSend * count, type, preRingPeer, comm, stream));
		NCCLCHECK(ncclGroupEnd());

		// cudaStreamSynchronize(stream);

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

		// cudaStreamSynchronize(s_recv);

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

testResult_t ncclSendRecv(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                          cudaStream_t stream, int rank, int nRanks, int peer) {
	// if (rank == 0 || rank == 8) {
	NCCLCHECK(ncclGroupStart());
	NCCLCHECK(ncclSend(sendbuff, count, type, peer, comm, stream));
	NCCLCHECK(ncclRecv(recvbuff, count, type, peer, comm, stream));
	NCCLCHECK(ncclGroupEnd());
	CUDACHECK(cudaStreamSynchronize(stream));
	// }

	return testSuccess;
}

void cal_intra_comm(int rank, int nRank, int* p2p) {
	for (int i = 0; i < nRank; i++) {
		if ((rank < 8 && i < 8) || (rank >= 8 && i >= 8))
			p2p[i] = 1;
	}
}

int p2p[1024];
testResult_t ncclAlltoAll_KDK(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclRedOp_t op,
                              int root, ncclComm_t comm, cudaStream_t stream, int rank, int nRanks, void* tmpSendbuffs,
                              void* tmpRecvbuffs, cudaStream_t stream_copy) {
	int ib_count = 0;
	int local_rank = rank % 8;
	int ib_peer = (rank + 8) % nRanks;
	size_t rankOffset = count * wordSize(type);
	// log("Rank-[%d], ib_peer-[%d]", rank, ib_peer);

	for (int i = 0; i < nRanks; i++) {
		if ((rank < 8 && i < 8) || (rank >= 8 && i >= 8))
			p2p[i] = 1;
	}

	for (int r = 0; r < nRanks; r++) {
		if (!p2p[r]) {
			// log("Rank-[%d], inter-peer-[%d]", rank, r);
			CUDACHECK(cudaMemcpyAsync(((char*)tmpSendbuffs) + ib_count * rankOffset, ((char*)sendbuff) + r * rankOffset,
			                          rankOffset, cudaMemcpyDeviceToDevice, stream));
			ib_count++;
		}
	}

	if (local_rank & 1) {
		NCCLCHECK(ncclGroupStart());
		for (int r = 0; r < nRanks; r++) {
			if (p2p[r]) {
				// log("Rank-[%d], intra-peer-[%d]", rank, r);
				NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream_copy));
				NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream_copy));
			}
		}
		NCCLCHECK(ncclGroupEnd());

		NCCLCHECK(ncclGroupStart());
		NCCLCHECK(ncclSend(tmpSendbuffs, ib_count * count, type, ib_peer, comm, stream));
		NCCLCHECK(ncclRecv(tmpRecvbuffs, ib_count * count, type, ib_peer, comm, stream));
		NCCLCHECK(ncclGroupEnd());

		NCCLCHECK(ncclGroupStart());
		ib_count = 0;
		for (int r = 0; r < nRanks; r++) {
			if (p2p[r]) {
				// log("Rank-[%d], block-[%d], send to intra-peer-[%d], ", rank, r, r);
				NCCLCHECK(ncclSend(((char*)tmpRecvbuffs) + ib_count * rankOffset, count, type, r, comm, stream));
				ib_count++;
			} else {
				int local_midd_ib_peer = (r + 8) % nRanks;
				// log("Rank-[%d], block-[%d], recv from local_midd_ib_peer-[%d], ", rank, r, local_midd_ib_peer);
				NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, local_midd_ib_peer, comm, stream));
			}
		}
		NCCLCHECK(ncclGroupEnd());
	} else {
		// cudaStreamSynchronize(stream);
		NCCLCHECK(ncclGroupStart());
		NCCLCHECK(ncclSend(tmpSendbuffs, ib_count * count, type, ib_peer, comm, stream));
		NCCLCHECK(ncclRecv(tmpRecvbuffs, ib_count * count, type, ib_peer, comm, stream));
		NCCLCHECK(ncclGroupEnd());

		NCCLCHECK(ncclGroupStart());
		for (int r = 0; r < nRanks; r++) {
			if (p2p[r]) {
				// log("Rank-[%d], intra-peer-[%d]", rank, r);
				NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream_copy));
				NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream_copy));
			}
		}
		NCCLCHECK(ncclGroupEnd());

		ib_count = 0;
		NCCLCHECK(ncclGroupStart());
		for (int r = 0; r < nRanks; r++) {
			if (p2p[r]) {
				// log("Rank-[%d], block-[%d], send to intra-peer-[%d], ", rank, r, r);
				NCCLCHECK(ncclSend(((char*)tmpRecvbuffs) + ib_count * rankOffset, count, type, r, comm, stream));
				ib_count++;
			} else {
				int local_midd_ib_peer = (r + 8) % nRanks;
				// log("Rank-[%d], block-[%d], recv from local_midd_ib_peer-[%d], ", rank, r, local_midd_ib_peer);
				NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, local_midd_ib_peer, comm, stream));
			}
		}
		NCCLCHECK(ncclGroupEnd());
	}
	return testSuccess;
}

testResult_t ncclAlltoAll_IntraInter(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                                     cudaStream_t stream, int rank, int nRanks) {
	NCCLCHECK(ncclCommCount(comm, &nRanks));
	size_t rankOffset = count * wordSize(type);
	bool local_flag = (bool)(rank & 1);

	for (int i = 0; i < nRanks; i++) {
		if ((rank < 8 && i < 8) || (rank >= 8 && i >= 8))
			p2p[i] = 1;
	}

	NCCLCHECK(ncclGroupStart());
	for (int r = 0; r < nRanks; r++) {
		if ((p2p[r] && !(r & 1) && !local_flag) || (!p2p[r] && !(r & 1) && !local_flag)) {
			NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
			NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
		}
	}
	NCCLCHECK(ncclGroupEnd());

	cudaStreamSynchronize(stream);

	NCCLCHECK(ncclGroupStart());
	for (int r = 0; r < nRanks; r++) {
		if ((p2p[r] && (r & 1) && !local_flag) || (!p2p[r] && (r & 1) && !local_flag)) {
			NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
			NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
		}
	}
	NCCLCHECK(ncclGroupEnd());

	cudaStreamSynchronize(stream);

	NCCLCHECK(ncclGroupStart());
	for (int r = 0; r < nRanks; r++) {
		if ((p2p[r] && !(r & 1) && local_flag) || (!p2p[r] && (r & 1) && !local_flag)) {
			NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
			NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
		}
	}
	NCCLCHECK(ncclGroupEnd());

	cudaStreamSynchronize(stream);

	NCCLCHECK(ncclGroupStart());
	for (int r = 0; r < nRanks; r++) {
		if ((!p2p[r] && !(r & 1) && local_flag) || (p2p[r] && (r & 1) && !local_flag)) {
			NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
			NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
		}
	}
	NCCLCHECK(ncclGroupEnd());

	cudaStreamSynchronize(stream);

	return testSuccess;
}

testResult_t ncclAlltoAll_Sync2(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                                cudaStream_t stream, int rank, int nRanks) {
	NCCLCHECK(ncclCommCount(comm, &nRanks));
	size_t rankOffset = count * wordSize(type);
	bool small_rank = rank >= 8 ? false : true;
	// int remote_bound = small_rank ? 0 : 8;
	int local_bound = small_rank ? 0 : 8;
	// int now = 0, steps = 1;
	// int next = steps;
	// int mirror_rank = (rank + 8) % 16;
	// int mirror_local_rank = (rank + 8) % 8;
	int remote_bound = (rank + 8) % 16;

	// NCCLCHECK(ncclGroupStart());
	// for (int r = local_bound; r < local_bound + 8; r++) {
	// 	NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
	// 	NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
	// }
	// NCCLCHECK(ncclGroupEnd());

	// if (rank & 1)
	// 	cudaStreamSynchronize(stream);

	for (int now = 0; now < 8; now++) {
		int local_peer_send = (local_bound + now) % 16;
		int local_peer_recv = (local_bound - now + 16) % 16;

		if (small_rank) {
			int peer = (remote_bound + now) % 16;
			if (peer < 8)
				peer = peer + 8;
			if (local_peer_send > 8)
				local_peer_send = local_peer_send - 8;
			if (local_peer_recv > 8)
				local_peer_recv = local_peer_recv - 8;

			NCCLCHECK(ncclGroupStart());
			if (local_peer_send != rank) {
				NCCLCHECK(ncclSend(((char*)sendbuff) + local_peer_send * rankOffset, count, type, local_peer_send, comm,
				                   stream));
				NCCLCHECK(ncclRecv(((char*)recvbuff) + local_peer_recv * rankOffset, count, type, local_peer_recv, comm,
				                   stream));
			}
			NCCLCHECK(ncclGroupEnd());

			NCCLCHECK(ncclGroupStart());
			// log("rank:[%d], now:[%d], peer:[%d]", rank, now, peer);
			NCCLCHECK(ncclRecv(((char*)recvbuff) + peer * rankOffset, count, type, peer, comm, stream));
			NCCLCHECK(ncclSend(((char*)sendbuff) + peer * rankOffset, count, type, peer, comm, stream));
			NCCLCHECK(ncclGroupEnd());

		} else {
			int peer = (16 + remote_bound - now) % 16;
			if (peer > 8)
				peer = peer - 8;
			if (local_peer_send < 8)
				local_peer_send = local_peer_send + 8;
			if (local_peer_recv < 8)
				local_peer_recv = local_peer_recv + 8;

			NCCLCHECK(ncclGroupStart());
			if (local_peer_send != rank) {
				NCCLCHECK(ncclSend(((char*)sendbuff) + local_peer_send * rankOffset, count, type, local_peer_send, comm,
				                   stream));
				NCCLCHECK(ncclRecv(((char*)recvbuff) + local_peer_recv * rankOffset, count, type, local_peer_recv, comm,
				                   stream));
			}
			NCCLCHECK(ncclGroupEnd());

			NCCLCHECK(ncclGroupStart());
			// log("rank:[%d], now:[%d], peer:[%d]", rank, now, peer);
			NCCLCHECK(ncclRecv(((char*)recvbuff) + peer * rankOffset, count, type, peer, comm, stream));
			NCCLCHECK(ncclSend(((char*)sendbuff) + peer * rankOffset, count, type, peer, comm, stream));
			NCCLCHECK(ncclGroupEnd());
		}

		// if (now & 1)
		// 	cudaStreamSynchronize(stream);
	}

	cudaStreamSynchronize(stream);
	return testSuccess;
}

testResult_t ncclAlltoAll_Sync3(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                                cudaStream_t stream, int rank, int nRanks) {
	// NCCLCHECK(ncclCommCount(comm, &nRanks));
	// size_t rankOffset = count * wordSize(type);
	// bool small_rank = rank >= 8 ? false : true;
	// // int remote_bound = small_rank ? 0 : 8;
	// int local_bound = small_rank ? 0 : 8;
	// // int now = 0, steps = 1;
	// // int next = steps;
	// // int mirror_rank = (rank + 8) % 16;
	// // int mirror_local_rank = (rank + 8) % 8;
	// int remote_bound = (rank + 8) % 16;

	// cudaStreamSynchronize(stream);
	return testSuccess;
}

testResult_t ncclAlltoAll_gatherAndSend(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type,
                                        ncclComm_t comm, cudaStream_t stream, int rank, int nRanks) {
	NCCLCHECK(ncclCommCount(comm, &nRanks));
	size_t rankOffset = count * wordSize(type);
	bool local_flag = (bool)(rank & 1);

	for (int i = 0; i < nRanks; i++) {
		if ((rank < 8 && i < 8) || (rank >= 8 && i >= 8))
			p2p[i] = 1;
	}

	NCCLCHECK(ncclGroupStart());
	for (int r = 0; r < nRanks; r++) {
		if ((p2p[r] && !(r & 1) && !local_flag) || (!p2p[r] && !(r & 1) && !local_flag)) {
			NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
			NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
		}
	}
	NCCLCHECK(ncclGroupEnd());

	return testSuccess;
}

testResult_t ncclAlltoAll(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                          cudaStream_t stream) {
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

int all2AllBruck(int rank, int nRanks, int localRank, size_t msgSize, size_t all2allSize, double nic_flow_size) {
	// NCCL and CUDA stuff
	int n, i;
	ncclUniqueId id;
	ncclComm_t comm;
	cudaStream_t s;
	cudaStream_t s_send, s_recv;
	BRUCK_TYPE *sendbuff, *recvbuff, *tmpBuffer, *syncSendBuff, *syncRecvBuff;
	size_t count = msgSize / sizeof(BRUCK_TYPE);
	struct timespec time1, time2;
	double diffInNanos = 0;

	// generating NCCL unique ID at one process and broadcasting it to all
	if (rank == 0) {
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
	CUDACHECK(cudaSetDevice(localRank));
	CUDACHECK(cudaStreamCreate(&s));
	CUDACHECK(cudaStreamCreate(&s_send));
	CUDACHECK(cudaStreamCreate(&s_recv));
	// Init data.
	initData(&sendbuff, &recvbuff, &tmpBuffer, &syncSendBuff, &syncRecvBuff, rank, all2allSize, count, false);
	NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, rank));

	// REPEAT
	for (int j = 0; j < 10; j++) {
		sock_barrier(nRanks, rank);
		ncclAlltoAll(sendbuff, recvbuff, count, ncclFloat, comm, s);
		cudaStreamSynchronize(s);
		initData(&sendbuff, &recvbuff, &tmpBuffer, &syncSendBuff, &syncRecvBuff, rank, all2allSize, count, true);
	}

	// REPEAT
	// BRUCK_NCCL, ONE2ONE_NCCL, INTRAINTER_NCCL, INTRAINTER_SYNC2_NCCL
	bool do_all2all = true;
	enum All2AllType all2all_T = 10;
	if (do_all2all) {
		void* tmpSendBuff = tmpBuffer;
		void* tmpRecvBuffer = tmpBuffer + (count * nRanks / 2);
		log("All2AllType: %d", all2all_T);
		for (int j = 0; j < REPEAT; j++) {
			sock_barrier(nRanks, rank);
			clock_gettime(CLOCK_MONOTONIC, &time1);
			switch (all2all_T) {
			case BRUCK_NCCL:
				bruckNCCL(sendbuff, recvbuff, tmpBuffer, count, ncclFloat, (ncclRedOp_t)0, 0, comm, s, false, j, s_send,
				          s_recv);
				cudaStreamSynchronize(s);
				cudaStreamSynchronize(s_send);
				break;
			case ONE2ONE_NCCL:
				ncclAlltoAll_KDK(sendbuff, recvbuff, count, ncclFloat, (ncclRedOp_t)0, 0, comm, s, rank, nRanks,
				                 tmpSendBuff, tmpRecvBuffer, s_send);
				cudaStreamSynchronize(s);
				cudaStreamSynchronize(s_send);
				break;
			case INTRAINTER_NCCL:
				ncclAlltoAll_IntraInter(sendbuff, recvbuff, count, ncclFloat, comm, s, rank, nRanks);
				break;
			case INTRAINTER_SYNC2_NCCL:
				ncclAlltoAll_Sync2(sendbuff, recvbuff, count, ncclFloat, comm, s, rank, nRanks);
				break;
			default:
				ncclAlltoAll(sendbuff, recvbuff, count, ncclFloat, comm, s);
				cudaStreamSynchronize(s);
				break;
			}
			clock_gettime(CLOCK_MONOTONIC, &time2);
			diffInNanos += (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);

			if (j == REPEAT - 1 && all2all_T != BRUCK_NCCL)
				cudaMemcpy((void*)tmpBuffer, (void*)recvbuff, count * nRanks * wordSize(ncclFloat),
				           cudaMemcpyDeviceToDevice);

			initData(&sendbuff, &recvbuff, &tmpBuffer, &syncSendBuff, &syncRecvBuff, rank, all2allSize, count, true);
		}

		log("Rank-[%d], msg_size:[%ld] MB,all2allSize:[%ld] MB, nic_flow_size:[%3.lf] MB avg_time:[%3.lf]", rank,
		    msgSize / (1024 * 1024), all2allSize / (1024 * 1024), nic_flow_size / (1024 * 1024),
		    (double)(diffInNanos) / (1e3 * REPEAT));

		sock_barrier(nRanks, rank);
		ncclAlltoAll(sendbuff, recvbuff, count, ncclFloat, comm, s);
		cudaStreamSynchronize(s);

		if (cmpAll2AllResult(tmpBuffer, recvbuff, nRanks, count, rank)) {
			log("Bruck all2all result is success!");
		} else {
			log("Bruck all2all result is failed!");
		}
	} else {
		size_t p2p_count = msgSize / sizeof(BRUCK_TYPE);
		// double diffInNanosList[32];
		// memset(diffInNanosList, 0, sizeof(double) * 32);
		diffInNanos = 0;

		for (int j = 0; j < REPEAT; j++) {
			// int remote_peer = ((rank + 8) % 16) + k;
			sock_barrier(nRanks, rank);
			if (rank == 0 || rank == 8) {
				int remote_peer = (rank == 0) ? 8 : 0;
				clock_gettime(CLOCK_MONOTONIC, &time1);
				ncclSendRecv(sendbuff, recvbuff, p2p_count, ncclFloat, comm, s, rank, nRanks, remote_peer);
				clock_gettime(CLOCK_MONOTONIC, &time2);
				diffInNanos += (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);
			}
		}
		log("Rank-[%d], p2p size:[%ld] MB, avg_time:[%3.lf]", rank, msgSize / (1024 * 1024),
		    (double)(diffInNanos) / (1e3 * REPEAT));
	}
error:
	return 0;
}
