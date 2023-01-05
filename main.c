#define _GNU_SOURCE /* See feature_test_macros(7) */

#include "bruck.h"
#include "client.h"
#include "config.h"
#include "debug.h"
#include "ib.h"
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

int local_node;
int node_num;

long max_time;
long min_time;
long avg_time;

struct cpuInfo node_list[SOCKET_NUMS];
FILE* log_fp = NULL;
pthread_t* send_thread;
pthread_t ibv_polling_t;

struct ConfigInfo config_info;
int init_env();
void destroy_env();

// #define CACHE_LINE_SIZE 64
// struct flag_cacheline {
// 	char pad[CACHE_LINE_SIZE - 1];
// 	char flag;
// }
int cal_numa_node(int locak_rank, int nRanks, int task_per_node) {
	if (USE_MULIT_THREAD) {
		int thread_per_rank = (nRanks - 1) + 2;
		int cpu_per_node = node_list[0].numa_cpu_num;
		return (thread_per_rank * locak_rank / cpu_per_node) % node_num;
	} else
		return 0;
}

int get_cpu_mask(void) {
	int ret, node = 0;

	node_num = numa_num_configured_nodes();
	CHECK(node_num != -1, "Failed to run numa_num_configured_nodes");
	struct bitmask* numa_mask = numa_allocate_cpumask();
	CHECK(numa_mask != NULL, "Failed to run numa_node_to_cpus");
	memset(node_list, 0, SOCKET_NUMS * sizeof(struct cpuInfo));

	for (node = 0; node < node_num; node++) {
		int long_int_num = 0;
		int core = 0;
		unsigned long bit_size;

		ret = numa_node_to_cpus(node, numa_mask);
		CHECK(ret == 0, "Failed to run numa_node_to_cpus");
		bit_size = numa_mask->size;

		// printf("CPU on Node-[%d]: ", node);
		while (bit_size != 0) {

			unsigned long cpu_mask = *(numa_mask->maskp + long_int_num);
			unsigned long mask = 1; // flag = (unsigned long)1 << (sizeof(unsigned long) * 8 - 1),
			while (true) {
				if (mask & cpu_mask) {
					// printf("%d, ", core);
					node_list[node].cpu_numa_list[node_list[node].numa_cpu_num++] = core;
				}

				core++;
				mask = mask << 1;
				if (mask == 0)
					break;
			}
			bit_size -= (sizeof(unsigned long) * 8);
			long_int_num++;
		}
		// printf("\n");
	}

	numa_free_cpumask(numa_mask);

	return 0;
error:
	return -1;
}

void do_setaffinity(int tid, int cpu) {
	cpu_set_t mask;
	CPU_ZERO(&mask);
	int rank = config_info.rank;
	int nRanks = config_info.nRanks;
	int task_per_node = config_info.task_per_node;
	int locak_rank = rank % task_per_node;

	if (local_node == -1)
		local_node = cal_numa_node(locak_rank, nRanks, task_per_node);

	if (cpu == -1)
		cpu = get_cpu_for_rank(locak_rank, nRanks, tid, local_node);

	CPU_SET(cpu, &mask);
	if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
		log_warn("warning: could not set CPU affinity, continuing...\n");
	}
	log("Rank-[%d], tid-[%d], set on Node-[%d]'s cpu-[%d]", rank, tid, local_node, cpu);
}

void* send_thread_func(void* aargs) {
	int peer_rank = (int)(int64_t)aargs;
	int rank = config_info.rank;
	int tid = (peer_rank > rank) ? (peer_rank - 1) : peer_rank;
	int send_count = 0;
	do_setaffinity(tid + SENDING_THREAD_ID, -1);

	struct ibv_qp* qp = ib_res.qp[peer_rank];
	size_t msg_size = config_info.msg_size;
	uint32_t lkey = ib_res.send_mr_info[0].mr->lkey;
	uint32_t rkey = ib_res.remote_mr_info[peer_rank][0].rkey;

	// volatile
	size_t send_offset = peer_rank * msg_size * config_info.num_concurr_msgs;
	size_t recv_offset = rank * msg_size * config_info.num_concurr_msgs;
	char* send_addr = ib_res.send_mr_info[0].addr + send_offset;
	char* remote_addr = (char*)(ib_res.remote_mr_info[peer_rank][0].addr) + recv_offset;

	while (send_count < REPEAT_TIME) {
		memory_barrier();
		while (unlikely(*(volatile char*)(send_addr) != (char)1))
			_mm_pause();

		// uint32_t imm = MSG_CTL_STOP;
		post_write(msg_size, lkey, (uint64_t)send_addr, qp, (char*)send_addr, rkey, remote_addr);
		// post_write_with_imm(msg_size, lkey, (uint64_t)send_addr, imm, qp, (char*)send_addr, rkey, remote_addr);

		memory_barrier();
		*send_addr = (char)0;
		send_count++;
	}

	return (void*)0;
}

long launch_send_multi_thread(int nRanks, int rank, int msg_size) {
	int i;
	struct timespec time1, time2;
	long diffInNanos;

	char* send_addr = ib_res.send_mr_info[0].addr;
	char* recv_addr = ib_res.recv_mr_info[0].addr + msg_size;

	// debug("send_addr = %" PRIx64 "", (uint64_t)send_addr);
	sock_barrier(nRanks, rank);
	clock_gettime(CLOCK_MONOTONIC, &time1);
	for (i = 0; i < nRanks; i++) {
		if (i != rank) {
			memory_barrier();
			*(volatile char*)(send_addr + msg_size - 1) = (char)1;
			memory_barrier();
			*(volatile char*)(send_addr) = (char)1;
		}
		send_addr += msg_size;
	}

	for (i = 0; i < nRanks; i++) {
		if (i != rank) {
			while (unlikely(*(volatile char*)(recv_addr - 1) != (char)1))
				_mm_pause();

			memory_barrier();
			*(recv_addr - 1) = (char)0;
		}
		recv_addr += msg_size;
	}

	memory_barrier();
	while (*send_all_flag == (uint8_t)0)
		_mm_pause();

	clock_gettime(CLOCK_MONOTONIC, &time2);
	diffInNanos = (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);
	// log("Rank-[%d] wait send_all_flag Use time [%.3lf] us", rank, (double)diffInNanos / 1000);
	return diffInNanos;
}

long launch_send_single_thread(int nRanks, int rank, int msg_size) {
	int peer, n, i, send_count = 0;
	int num_wc = nRanks - 1;
	struct timespec time1, time2;
	long diffInNanos = -1;

	char* send_addr = ib_res.send_mr_info[0].addr;
	char* recv_addr = ib_res.recv_mr_info[0].addr + msg_size;
	size_t remote_offset = rank * msg_size * config_info.num_concurr_msgs;

	struct ibv_cq* cq = ib_res.cq;
	struct ibv_wc* wc = NULL;
	uint32_t lkey = ib_res.send_mr_info[0].mr->lkey;

	// malloc cq.
	wc = (struct ibv_wc*)calloc_numa(num_wc * sizeof(struct ibv_wc));
	CHECK(wc != NULL, "failed to allocate wc.");

	// debug("send_addr = %" PRIx64 "", (uint64_t)send_addr);
	sock_barrier(nRanks, rank);
	clock_gettime(CLOCK_MONOTONIC, &time1);

	// init all send
	for (peer = 0; peer < nRanks; peer++) {
		struct ibv_send_wr send_wr, *bad_wr = NULL;

		if (peer != rank) {
			struct ibv_sge sge;
			struct ibv_qp* qp = ib_res.qp[peer];
			uint32_t rkey = ib_res.remote_mr_info[peer][0].rkey;
			char* remote_addr = ib_res.remote_mr_info[peer][0].addr + remote_offset;

			// set recv done flag.
			*(send_addr + msg_size - 1) = (char)1;

			memset(&send_wr, 0, sizeof(struct ibv_send_wr));
			send_wr.opcode = IBV_WR_RDMA_WRITE;
			send_wr.wr_id = (uint64_t)send_addr;
			send_wr.sg_list = &sge;
			send_wr.num_sge = 1;
			send_wr.send_flags = IBV_SEND_SIGNALED;
			send_wr.wr.rdma.remote_addr = (uintptr_t)remote_addr; // WGT
			send_wr.wr.rdma.rkey = rkey;

			sge.addr = (uintptr_t)send_addr;
			sge.length = msg_size;
			sge.lkey = lkey;

			CHECK(ibv_post_send(qp, &send_wr, &bad_wr) == 0, "ibv_post_send failed!");
		}
		send_addr += (msg_size * config_info.num_concurr_msgs);
	}

	// recving all data
	for (peer = 0; peer < nRanks; peer++) {
		if (peer != rank) {
			while (unlikely(*(volatile char*)(recv_addr - 1) != (char)1))
				_mm_pause();

			memory_barrier();
			*(recv_addr - 1) = (char)0;
		}
		recv_addr += msg_size;
	}

	// wait all send finish.
	while (likely(send_count != num_wc)) {
		do {
			n = ibv_poll_cq(cq, num_wc, wc);
		} while (n < 1);
		CHECK(n > 0, "failed to poll cq");

		for (i = 0; i < n; i++) {
			if (likely(wc[i].opcode == IBV_WC_RDMA_WRITE)) {
				send_count++;
			} else
				log("Fail polling wc");
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &time2);
	diffInNanos = (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);
	// log("Rank-[%d] wait send_all_flag Use time [%.3lf] us", rank, (double)diffInNanos / 1000);
error:
	return diffInNanos;
}

long IBSendRecvP2P(int nRanks, int rank, char* send_buff, char* recv_buff, int dst_peer, int src_peer, int msg_size) {
	int n;
	// int send_count = 0;
	int num_wc = 1;
	// struct timespec time1, time2;
	long diffInNanos = -1;
	struct ibv_wc wc;

	// size_t remote_offset = rank * msg_size * config_info.num_concurr_msgs;
	char* send_dst_addr = (char*)ib_res.remote_mr_info[dst_peer][0].addr;
	*(send_buff + msg_size - 1) = (char)1;

	// clock_gettime(CLOCK_MONOTONIC, &time1);
	CHECK(dst_peer != rank || src_peer != rank, "dst rank or src rank error");

	post_write(msg_size, ib_res.send_mr_info[0].mr->lkey, (uint64_t)send_buff, ib_res.qp[dst_peer], send_buff,
	           ib_res.remote_mr_info[dst_peer][0].rkey, send_dst_addr);

	while (unlikely(*(volatile char*)(recv_buff - 1) != (char)1))
		_mm_pause();

	memory_barrier();
	*(recv_buff - 1) = (char)0;

	// wait all send finish.
	do {
		n = ibv_poll_cq(ib_res.cq, num_wc, &wc);
	} while (n < 1);
	CHECK(n == 1, "failed to poll cq");
	CHECK(wc.opcode == IBV_WC_RDMA_WRITE, "Fail polling wc");

	// clock_gettime(CLOCK_MONOTONIC, &time2);
	// diffInNanos = (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);
	// log("Rank-[%d] wait send_all_flag Use time [%.3lf] us", rank, (double)diffInNanos / 1000);
error:
	return diffInNanos;
}

void launch_polling_and_sending_thread(int rank, int nRanks) {
	int ret, i;
	send_thread = (pthread_t*)calloc_numa(nRanks * sizeof(pthread_t));

	// launch ibv polling thread
	ret = pthread_create(&ibv_polling_t, NULL, pollingFunc, NULL);
	CHECK(ret == 0, "pollingFunc thread create error");

	// launch ibv sendding thread
	for (i = 0; i < nRanks; i++) {
		if (i != rank)
			ret = pthread_create(&send_thread[i], NULL, send_thread_func, (void*)(uint64_t)i);
		CHECK(ret == 0, "send_thread_func thread create error");
	}
error:
	return;
}

// struct ibv_mr* (*ibv_internal_reg_mr_iova2)(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova, int access);

int main(int argc, char* argv[]) {
	bool nccl_test = true;
	int ret = 0;
	int nDevs = 8;

	local_node = -1;
	config_info.num_concurr_msgs = 1;
	config_info.nRanks = atoi(argv[1]);
	config_info.nPeers = config_info.nRanks - 1;
	config_info.rank = atoi(argv[2]);
	config_info.msg_size = atoi(argv[3]);
	config_info.task_per_node = atoi(argv[4]);
	// config_info.base = atoi(argv[5]);
	// config_info.msg_size = 16;

	CHECK(init_env() == 0, "Failed to init env");

	if (nccl_test) {
		local_node = 0;
	} else {
		get_cpu_mask();
		do_setaffinity(MAIN_THREAD_ID, -1);
	}

	// return 0;

	config_info.sock_port_list = (char**)calloc_numa(config_info.nPeers * sizeof(char*));
	config_info.node_ip_list = (char**)calloc_numa(config_info.nPeers * sizeof(char*));

	for (int i = 0; i < config_info.nRanks; i++) {
		config_info.sock_port_list[i] = (char*)calloc_numa(8);
		config_info.node_ip_list[i] = (char*)calloc_numa(128);

		sprintf(config_info.sock_port_list[i], "%d", 13510 + i);
		sprintf(config_info.node_ip_list[i], "%s", argv[5 + i]);
	}

	int i = 0, rank = config_info.rank, nRanks = config_info.nRanks;
	size_t msg_size = config_info.msg_size;
	size_t all2all_size = msg_size * nRanks;
	double nic_flow_size = nccl_test ? ((double)8.0 * msg_size * nRanks / 2) : ((double)8.0 * msg_size * (nRanks - 1));

	recv_all_flag = (uint8_t*)calloc_numa(1);
	polling_flag = (uint8_t*)calloc_numa(1);
	send_all_flag = (uint8_t*)calloc_numa(1);

	*recv_all_flag = 0;
	*send_all_flag = 0;
	*polling_flag = 1;

	/* connect QP */
	CHECK(sock_handshack(nRanks, rank) == 0, "sock_handshack  error");
	CHECK(setup_ib(config_info.nRanks) == 0, "Failed to setup IB");
	CHECK(setup_ib_buffer(nRanks, nDevs) == 0, "Setip ib");

	if (!nccl_test) {
		CHECK(alloc_ib_buffer(nRanks, &(ib_res.send_mr_info[0])) == 0, "alloc ib buffer");
		CHECK(alloc_ib_buffer(nRanks, &(ib_res.recv_mr_info[0])) == 0, "alloc ib buffer");
		if (USE_MULIT_THREAD)
			launch_polling_and_sending_thread(rank, nRanks);

		// size_t buf_size = ib_res.ib_buf_size;
		log("Start warnup!");
		for (i = 0; i < WARMUP; i++) {
			sock_barrier(nRanks, rank);
			if (USE_MULIT_THREAD)
				launch_send_multi_thread(nRanks, rank, msg_size);
			else
				launch_send_single_thread(nRanks, rank, msg_size);
		}

		long max_time = 0;
		long min_time = INT64_MAX;
		long avg_time = 0;
		long tt;
		int max_idx = 0, min_idx = 0;
		int range[10];
		memset(range, 0, sizeof(int) * 10);

		log("Start repeat!");
		for (i = 0; i < REPEAT; i++) {
			sock_barrier(nRanks, rank);
			if (USE_MULIT_THREAD)
				tt = launch_send_multi_thread(nRanks, rank, msg_size);
			else
				tt = launch_send_single_thread(nRanks, rank, msg_size);

			if (tt > max_time) {
				max_time = tt;
				max_idx = i;
			}

			if (tt < min_time) {
				min_time = tt;
				min_idx = i;
			}

			avg_time += tt;
			// log("Rank-[%d], round-[%d] time [%.3lf] us", rank, i, (double)tt / 1000);
			range[(int)(tt / (long)1e6) / 10]++;
			sock_barrier(nRanks, rank);
		}

		// usleep(1000);
		log("Rank-[%d], msg_size:[%ld], all2allSize:[%ld] MB, nic_flow_size:[%.3lf] MB, max_time:[%.3lf]us,idx-[%d], "
		    "min_time:[%.3lf]us, idx-[%d], avg_time:[%.3lf]us",
		    rank, msg_size, all2all_size / (1024 * 1024), nic_flow_size / (1024 * 1024), (double)max_time / 1e3,
		    max_idx, (double)min_time / 1e3, min_idx, (double)avg_time / (1e3 * REPEAT));

		// for (i = 0; i < 10; i++)
		// 	log("Rank-[%d], range[%d ms-%d ms]:[%d]", rank, i * 10, (i + 1) * 10, range[i]);
		if (USE_MULIT_THREAD) {
			*polling_flag = 0;
			pthread_join(ibv_polling_t, NULL);
		}
	} else {
		// all2AllBruck(rank, nRanks, rank % config_info.task_per_node, msg_size, all2all_size, nic_flow_size);

		all2AllBruck_nGPUs(16, 8, msg_size, (rank == 0) ? 0 : 8, rank);
	}

	close_sock(nRanks);

	return 0;
error:
	// if (ibvhandle != NULL) dlclose(ibvhandle);
	if (!nccl_test)
		close_ib_connection();
	destroy_env();
	return ret;
}

void print_config_info() {
	log(LOG_SUB_HEADER, "Configuraion");

	log("nRanks                    = %d", config_info.nRanks);
	log("rank                      = %d", config_info.rank);
	log("msg_size                  = %d", config_info.msg_size);
	for (int i = 0; i < config_info.nRanks; i++) {
		log("port                      = %s", config_info.sock_port_list[i]);
		log("Node                      = %s", config_info.node_ip_list[i]);
	}

	// log("num_concurr_msgs          = %d", config_info.num_concurr_msgs);
	// log("sock_port                 = %s", config_info.sock_port);
	// log("server name               = %s", config_info.servers[0]);
	// log("client name               = %s", config_info.clients[0]);

	log(LOG_SUB_HEADER, "End of Configuraion");
}

int init_env() {
	char fname[64] = {'\0'};
	const char* mulit_threads = USE_MULIT_THREAD ? "multi_theads" : "singe_thread";
	sprintf(fname, "[%s]-rank-[%d].log", mulit_threads, config_info.rank);

	log_fp = fopen(fname, "w");
	CHECK(log_fp != NULL, "Failed to open log file");

	// log(LOG_HEADER, "IB Echo Server");
	// print_config_info();

	return 0;
error:
	return -1;
}

void destroy_env() {
	log(LOG_HEADER, "Run Finished");
	if (log_fp != NULL) {
		fclose(log_fp);
	}
}
