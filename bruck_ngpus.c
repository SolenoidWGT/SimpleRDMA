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

struct All2AllInfo {

	int nDevs;
	int socket_rank;
	int socket_nRanks;
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

	struct IBMemInfo* host_sendbuff;
	struct IBMemInfo* host_recvbuff;
	BRUCK_TYPE** dev_sendbuff;
	BRUCK_TYPE** dev_recvbuff;

	cudaStream_t* streams;
	ncclComm_t* comms;
};

int post_recv_wr(int socket_rank, int socket_nRanks, int nDevs, size_t block_size) {

	for (int dev = 0; dev < nDevs; dev++) {
		uint32_t lkey = ib_res.recv_mr_info[dev].mr->lkey;
		char* recv_ptr = ib_res.recv_mr_info[dev].addr;
		for (int rank = 0; rank < socket_nRanks; rank++) {
			if (socket_rank != rank) {
				int chunk_num = block_size / chunk_size;
				char* chunk_addr = recv_ptr;
				for (int ch = 0; ch < chunk_num; ch++) {
					CHECK(post_srq_recv(chunk_size, lkey, (uint64_t)chunk_addr, ib_res.srq, chunk_addr) == 0,
					      "post recv");
					chunk_addr += chunk_size;
				}
				recv_ptr += block_size;
			}
		}
	}

	return 0;
error:
	return -1;
}

void* launch_send_single_thread(void* arg) {
	// int send_peers = args->send_peers;
	// rank info
	do_setaffinity(POLLING_THREAD_ID, -1);

	struct ib_send_args* args = (struct ib_send_args*)arg;
	long diffInNanos = -1;
	int loop_count = 0;
	int loop_num = args->loop_num;
	int dev_nRanks = args->dev_nRanks;
	// int dev_rank = args->dev_rank;
	int socket_nRanks = args->socket_nRanks;
	int socket_rank = args->socket_rank;
	// mesage info
	int block_size = args->block_size;
	int chunk_size = args->chunk_size;
	bool use_pcie_relaxed_order = args->use_pcie_relaxed_order;
	bool need_barrier = args->need_barrier;
	bool only_send = args->only_send;
	bool imm_send = args->imm_send;
	struct IBMemInfo* send_mr_info = ib_res.send_mr_info;
	struct IBMemInfo* recv_mr_info = ib_res.recv_mr_info;
	struct MRinfo** remote_mr_info = ib_res.remote_mr_info;

	if (!imm_send && use_pcie_relaxed_order)
		log("Warring: Using imm_send and pcie_relaxed_order at the same time!");

	while (loop_count < loop_num) {
		struct timespec time1, time2;
		int num_wc = 0;
		if (need_barrier)
			sock_barrier(socket_nRanks, socket_rank);

		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
		// init all send
		for (int dev = 0; dev < dev_nRanks; dev++) {
			char* send_addr = send_mr_info[dev].addr;
			uint32_t lkey = send_mr_info[dev].mr->lkey;
			size_t remote_offset = dev * block_size;

			for (int peer = 0; peer < socket_nRanks; peer++) {
				if (peer != socket_rank) {
					struct ibv_qp* qp = ib_res.qp[peer];
					uint32_t rkey = remote_mr_info[peer][dev].rkey;
					char* remote_addr = remote_mr_info[peer][dev].addr + remote_offset;

					// set recv done flag.
					if (imm_send) {
						int chunk_nums = block_size / chunk_size;
						char* chunk_addr = send_addr;
						char* remote_chunk_addr = remote_addr;
						for (int ch = 0; ch < chunk_nums; ch++) {
							post_write_imm(chunk_size, lkey, (uint64_t)chunk_addr, qp, (char*)chunk_addr, rkey,
							               remote_chunk_addr, (uint32_t)1);
							if (only_send)
								num_wc += 1;
							else
								num_wc += 2;
							chunk_addr += chunk_size;
							remote_chunk_addr += chunk_size;
						}
					} else {
						*(send_addr + block_size - 1) = (char)1;
						post_write(block_size, lkey, (uint64_t)send_addr, qp, (char*)send_addr, rkey, remote_addr);
						num_wc++;
					}
				}
				send_addr += block_size;
			}
		}

		// // recving all data
		// if (!use_pcie_relaxed_order) {
		// 	for (int dev = 0; dev < dev_nRanks; dev++) {
		// 		char* recv_addr = recv_mr_info[dev].addr + block_size;
		// 		for (int peer = 0; peer < socket_nRanks; peer++) {
		// 			if (peer != socket_rank) {
		// 				while (unlikely(*(volatile char*)(recv_addr - 1) != (char)1))
		// 					_mm_pause();

		// 				memory_barrier();
		// 				*(recv_addr - 1) = (char)0;
		// 			}
		// 			recv_addr += block_size;
		// 		}
		// 	}
		// }

		// wait all send or recv finish.
		int n;
		int wc_count = 0, write_wc = 0, recv_wc = 0;
		struct ibv_wc* wc = ib_res.wc;
		while (likely(wc_count < num_wc)) {
			do {
				n = ibv_poll_cq(ib_res.cq, MAX_WC_NUMS, wc);
			} while (n < 1);
			CHECK(n > 0, "failed to poll cq");

			for (int i = 0; i < n; i++) {
				if (wc[i].opcode == IBV_WC_RDMA_WRITE) {
					wc_count++;
					write_wc++;
				} else if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
					wc_count++;
					recv_wc++;
				} else {
					log("Fail polling wc");
				}
			}
		}

		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
		diffInNanos = (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);
		loop_count++;
	}

	// log("Rank-[%d] wait send_all_flag Use time [%.3lf] us", rank, (double)diffInNanos / 1000);
error:
	return (void*)diffInNanos;
}

testResult_t initData_n(struct All2AllInfo* info) {

	// for (int i = 0; i < info->nDevs; i++) {
	// 	int dev_rank = info->base + i;
	// 	char* rank_i_addr = ib_res.send_mr_info[i].addr;
	// 	for (int j = 0; j < info->nRanks; j++) {
	// 		char* block_j_rank_i_addr = rank_i_addr + j * info->block_size;
	// 		memset(block_j_rank_i_addr, j + dev_rank, info->block_size);
	// 	}

	// 	CUDACHECK(
	// 	    cudaMemcpy(info->dev_sendbuff[i], ib_res.send_mr_info[i].addr, info->buff_size, cudaMemcpyHostToDevice));
	// 	memset(ib_res.send_mr_info[i].addr, 0, info->buff_size);
	// }

	return testSuccess;
}

int allocHostBuff(struct All2AllInfo* info, struct IBMemInfo* buff) {
	// struct IBMemInfo* buff = (struct IBMemInfo*)calloc_numa(info->nDevs * sizeof(struct IBMemInfo));

	for (int i = 0; i < info->nDevs; i++) {
		buff->mr_id = 0; // Every GPUs only has one buffer

		CUDACHECK(cudaSetDevice(i));
		CUDACHECK(cudaHostAlloc((void**)&(buff[i].addr), info->buff_size, cudaHostAllocDefault));
		memset(buff[i].addr, 0, info->buff_size);
		CHECK(register_ib_mr(buff[i].addr, info->buff_size, &(buff[i].mr), &(buff[i].mr_info)) == 0, "mr exchange");
		CHECK(exchange_mr_info(buff, false) == 0, "mr exchange");
	}

	return 0;
error:
	return -1;
}

testResult_t allocCudaBuff(struct All2AllInfo* info) {
	info->dev_sendbuff = (BRUCK_TYPE**)calloc_numa(info->nDevs * sizeof(BRUCK_TYPE*));
	info->dev_recvbuff = (BRUCK_TYPE**)calloc_numa(info->nDevs * sizeof(BRUCK_TYPE*));

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
	allocHostBuff(info, ib_res.recv_mr_info);
	allocHostBuff(info, ib_res.send_mr_info);
	info->host_sendbuff = ib_res.send_mr_info;
	info->host_recvbuff = ib_res.recv_mr_info;

	allocCudaBuff(info);
	// error:
	return testSuccess;
}

testResult_t ncclAlltoAll_n(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                            cudaStream_t stream, struct All2AllInfo* info) {
	size_t rankOffset = count * sizeof(BRUCK_TYPE);

	// NCCLCHECK(ncclGroupStart());
	for (int r = 0; r < info->nRanks; r++) {
		NCCLCHECK(ncclSend(((char*)sendbuff) + r * rankOffset, count, type, r, comm, stream));
		NCCLCHECK(ncclRecv(((char*)recvbuff) + r * rankOffset, count, type, r, comm, stream));
	}
	// NCCLCHECK(ncclGroupEnd());

	return testSuccess;
}

testResult_t ngpu_async_gpu_mem_DeviceToHost(struct All2AllInfo* info, enum cudaMemcpyKind copy_direction, int devs) {

	if (devs > info->nDevs)
		return testInternalError;

	for (int i = 0; i < devs; i++) {
		if (copy_direction == cudaMemcpyDeviceToHost)
			CUDACHECK(cudaMemcpyAsync(info->host_sendbuff[i].addr, info->dev_sendbuff[i], info->buff_size,
			                          copy_direction, info->streams[i]));
		else
			CUDACHECK(cudaMemcpyAsync(info->dev_sendbuff[i], info->host_sendbuff[i].addr, info->buff_size,
			                          copy_direction, info->streams[i]));
	}
	for (int i = 0; i < devs; i++) {
		CUDACHECK(cudaStreamSynchronize(info->streams[i]));
	}
	return testSuccess;
}

typedef testResult_t (*test_fn)(struct All2AllInfo*);
testResult_t test_cpy_d2h_8(struct All2AllInfo* info) {
	ngpu_async_gpu_mem_DeviceToHost(info, cudaMemcpyDeviceToHost, 8);
	return testSuccess;
}
testResult_t test_cpy_h2d_8(struct All2AllInfo* info) {
	ngpu_async_gpu_mem_DeviceToHost(info, cudaMemcpyHostToDevice, 8);
	return testSuccess;
}
testResult_t test_cpy_d2h_1(struct All2AllInfo* info) {
	ngpu_async_gpu_mem_DeviceToHost(info, cudaMemcpyDeviceToHost, 1);
	return testSuccess;
}
testResult_t test_cpy_h2d_1(struct All2AllInfo* info) {
	ngpu_async_gpu_mem_DeviceToHost(info, cudaMemcpyHostToDevice, 1);
	return testSuccess;
}
testResult_t test_all2all(struct All2AllInfo* info) {
	int finish_count = 0;
	NCCLCHECK(ncclGroupStart());
	for (int j = 0; j < info->nDevs; j++)
		ncclAlltoAll_n(info->dev_sendbuff[j], info->dev_recvbuff[j], info->count_per_block, ncclFloat, info->comms[j],
		               info->streams[j], info);
	NCCLCHECK(ncclGroupEnd());

	while (unlikely(finish_count < info->nDevs)) {
		for (int j = 0; j < info->nDevs; j++) {
			if (unlikely(cudaStreamQuery(info->streams[j]) == cudaSuccess))
				finish_count++;
			// cudaStreamSynchronize(info->streams[j]);
		}
		_mm_pause();
	}
	return testSuccess;
}

testResult_t test_ib_send_recv_async(struct All2AllInfo* info) {
	struct ib_send_args args;
	pthread_t ibv_proxy_t;

	args.loop_num = 10 * REPEAT;
	args.dev_nRanks = info->nDevs;
	args.socket_nRanks = info->socket_nRanks;
	args.socket_rank = info->socket_rank;
	args.block_size = info->block_size;
	args.chunk_size = info->block_size;
	args.use_pcie_relaxed_order = true;
	args.need_barrier = false;
	args.only_send = false;
	args.imm_send = false;

	if (args.imm_send) {
		post_recv_wr(args.socket_rank, args.socket_nRanks, args.dev_nRanks, args.block_size);
	}

	CHECK(pthread_create(&ibv_proxy_t, NULL, launch_send_single_thread, (void*)(&args)) == 0, "create ib thread");
	// CHECK(pthread_join(ibv_proxy_t, NULL) == 0, "join ib thread");
	return testSuccess;
error:
	return testInternalError;
}

testResult_t loop_engine(struct All2AllInfo* info, test_fn fn) {
	int socket_rank = info->socket_rank;
	struct timespec time1, time2;
	memset(info->max_times, 0, sizeof(long) * info->nDevs);
	memset(info->avg_times, 0, sizeof(long) * info->nDevs);
	for (int i = 0; i < info->nDevs; i++) {
		info->range[i] = (int*)calloc_numa(sizeof(int) * 10);
		info->min_times[i] = INT64_MAX;
	}
	// REPEAT
	for (int i = 0; i < REPEAT_TIME; i++) {
		sock_barrier(2, socket_rank);
		if (i == 5)
			log("running...");

		// Don't use CLOCK_PROCESS_CPUTIME_ID to monitor
		clock_gettime(CLOCK_MONOTONIC, &time1);
		(*fn)(info);
		clock_gettime(CLOCK_MONOTONIC, &time2);

		if (i >= WARMUP) {
			long tt = (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);

			if (tt > info->max_times[0]) {
				info->max_times[0] = tt;
				info->max_idxs[0] = (i - WARMUP);
			}

			if (tt < info->min_times[0]) {
				info->min_times[0] = tt;
				info->min_idxs[0] = (i - WARMUP);
			}

			info->avg_times[0] += tt;
			info->range[0][(int)(tt / (long)1e6) / 10]++;
		}

		initData_n(info);
	}

	// if (cmpAll2AllResult(tmpBuffer, recvbuff, nRanks, count, rank)) {
	// 	log("Bruck all2all result is success!");
	// } else {
	// 	log("Bruck all2all result is failed!");
	// }

	log("Base-[%d], nDevs-[%d], nRanks-[%d]", info->base, info->nDevs, info->nRanks);
	log("block:[%.3lf]MB, all2all:[%ld]MB", (double)info->block_size / MB, info->buff_size / MB);
	log("buffer:[%.3lf]MB, tbuffer:[%ld]MB", (double)info->buff_size / MB, info->total_buff_size / MB);
	log("max_time-[%.3lf]us,idx-[%d],min_time-[%.3lf]us, idx-[%d],avg_time:[%3.lf]us", (double)info->max_times[0] / 1e3,
	    info->max_idxs[0], (double)info->min_times[0] / 1e3, info->min_idxs[0],
	    (double)info->avg_times[0] / (1e3 * REPEAT));
	log("Max Bandwidth:[%lf] Gb/s",
	    ((double)(((double)info->buff_size / (128 * MB)) * 1e9) / (double)(info->min_times[0])));
	log("Avg Bandwidth:[%lf] Gb/s",
	    ((double)(((double)info->buff_size / (128 * MB)) * 1e9 * REPEAT) / (double)(info->avg_times[0])));
	return testSuccess;
}

int all2AllBruck_nGPUs(int nRanks, int nDevs, size_t block_size, int base, int rank, int peer_nRanks) {
	// NCCL and CUDA stuff
	struct All2AllInfo* info = (struct All2AllInfo*)calloc_numa(sizeof(struct All2AllInfo));
	ncclUniqueId id;

	info->nDevs = nDevs;
	info->socket_rank = rank;
	info->socket_nRanks = peer_nRanks;
	info->nRanks = nRanks;
	info->base = base;
	info->block_size = block_size;
	info->count_per_block = info->block_size / sizeof(BRUCK_TYPE);
	// info->count_per_rank = info->count_per_block * nRanks;
	info->buff_size = info->block_size * nRanks;
	info->total_buff_size = info->buff_size * nDevs;

	info->streams = (cudaStream_t*)calloc_numa(sizeof(cudaStream_t) * nDevs);
	info->comms = (ncclComm_t*)calloc_numa(sizeof(ncclComm_t) * nDevs);
	info->max_times = (long*)calloc_numa(sizeof(long) * nDevs);
	info->min_times = (long*)calloc_numa(sizeof(long) * nDevs);
	info->avg_times = (long*)calloc_numa(sizeof(long) * nDevs);
	// info->tts = (long*)calloc_numa(sizeof(long) * nDevs);
	info->max_idxs = (int*)calloc_numa(sizeof(int) * nDevs);
	info->min_idxs = (int*)calloc_numa(sizeof(int) * nDevs);
	info->range = (int**)calloc_numa(sizeof(int*) * nDevs);

	log("Base-[%d], nDevs-[%d], nRanks-[%d]", info->base, info->nDevs, info->nRanks);
	log("block:[%.3lf]MB, all2all:[%ld]MB", (double)info->block_size / (1024 * 1024), info->buff_size / (1024 * 1024));
	log("buffer:[%.3lf]MB, tbuffer:[%ld]MB", (double)info->buff_size / (1024 * 1024),
	    info->total_buff_size / (1024 * 1024));

	// generating NCCL unique ID at one process and broadcasting it to all
	if (info->base == 0) {
		ncclGetUniqueId(&id);
		for (int i = 1; i < info->socket_nRanks; i++) {
			CHECK(sock_write(peer_sockfd[i], (void*)&id, sizeof(ncclUniqueId)) == sizeof(ncclUniqueId),
			      "ncclUniqueId send error.");
		}
	} else {
		CHECK(sock_read(peer_sockfd[0], (void*)&id, sizeof(ncclUniqueId)) == sizeof(ncclUniqueId),
		      "ncclUniqueId recv error.");
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

	// test_fn cpy_d2h_fn = test_cpy_d2h_1;
	// test_fn cpy_h2d_fn = test_cpy_h2d_1;
	// test_fn all2all_fn = test_all2all;
	test_ib_send_recv_async(info);
	sock_barrier(info->socket_nRanks, info->socket_rank);

	loop_engine(info, test_cpy_d2h_1);
	loop_engine(info, test_cpy_h2d_1);
	// loop_engine(info, test_cpy_d2h_8);
	// loop_engine(info, test_cpy_h2d_8);

error:

	for (int i = 0; i < info->nDevs; i++) {
		ncclCommDestroy(info->comms[i]);
	}
	return 0;
}