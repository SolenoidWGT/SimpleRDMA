#define _GNU_SOURCE /* See feature_test_macros(7) */

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

int local_node;
int node_num;
struct cpuInfo node_list[SOCKET_NUMS];
FILE* log_fp = NULL;
pthread_t* send_thread;
struct ConfigInfo config_info;
int init_env();
void destroy_env();

// #define CACHE_LINE_SIZE 64
// struct flag_cacheline {
// 	char pad[CACHE_LINE_SIZE - 1];
// 	char flag;
// }
int cal_numa_node(int locak_rank, int nRanks, int task_per_node) {
	int thread_per_rank = (nRanks - 1) + 2;
	int cpu_per_node = node_list[0].numa_cpu_num;
	return (thread_per_rank * locak_rank / cpu_per_node) % node_num;
}

int get_cpu_mask_on_node0(void) {
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

		printf("CPU on Node-[%d]: ", node);
		while (bit_size != 0) {

			unsigned long cpu_mask = *(numa_mask->maskp + long_int_num);
			unsigned long mask = 1; // flag = (unsigned long)1 << (sizeof(unsigned long) * 8 - 1),
			while (true) {
				if (mask & cpu_mask) {
					printf("%d, ", core);
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
		printf("\n");
	}

	numa_free_cpumask(numa_mask);

	return 0;
error:
	return -1;
}

void do_setaffinity(int tid) {
	cpu_set_t mask;
	CPU_ZERO(&mask);
	int rank = config_info.rank;
	int nRanks = config_info.nRanks;
	int task_per_node = config_info.task_per_node;
	int locak_rank = rank % task_per_node;

	if (local_node == -1)
		local_node = cal_numa_node(locak_rank, nRanks, task_per_node);

	int cpu = get_cpu_for_rank(locak_rank, nRanks, tid, local_node);
	CPU_SET(cpu, &mask);
	if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
		printf("warning: could not set CPU affinity, continuing...\n");
	}
	printf("Rank-[%d], tid-[%d], set on Node-[%d]'s cpu-[%d]\n", rank, tid, local_node, cpu);
}

void* send_thread_func(void* aargs) {
	int peer_rank = (int)(int64_t)aargs;
	int rank = config_info.rank;
	int tid = (peer_rank > rank) ? (peer_rank - 1) : peer_rank;
	do_setaffinity(tid);

	struct ibv_qp* qp = ib_res.qp[peer_rank];
	size_t msg_size = config_info.msg_size;
	uint32_t lkey = ib_res.send_mr->lkey;
	uint32_t rkey = ib_res.remote_mr_info[peer_rank].rkey;

	// volatile
	size_t send_offset = peer_rank * msg_size * config_info.num_concurr_msgs;
	size_t recv_offset = rank * msg_size * config_info.num_concurr_msgs;
	char* send_addr = ib_res.ib_send_buf + send_offset;
	char* remote_addr = (char*)(ib_res.remote_mr_info[peer_rank].addr) + recv_offset;

	memory_barrier();
	while (unlikely(*(volatile char*)(send_addr) != (char)1))
		_mm_pause();

	// uint32_t imm = MSG_CTL_STOP;
	post_write(msg_size, lkey, (uint64_t)send_addr, qp, (char*)send_addr, rkey, remote_addr);
	// post_write_with_imm(msg_size, lkey, (uint64_t)send_addr, imm, qp, (char*)send_addr, rkey, remote_addr);
	return (void*)0;
}

// struct ibv_mr* (*ibv_internal_reg_mr_iova2)(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova, int access);

int main(int argc, char* argv[]) {
	int ret = 0, rank, nRanks, i;

	size_t msg_size;
	struct timespec time1, time2;
	long diffInNanos;

	local_node = -1;
	config_info.num_concurr_msgs = 1;
	config_info.nRanks = atoi(argv[1]);
	config_info.nPeers = config_info.nRanks - 1;
	config_info.rank = atoi(argv[2]);
	config_info.msg_size = atoi(argv[3]);
	config_info.task_per_node = atoi(argv[4]);
	// config_info.msg_size = 16;

	get_cpu_mask_on_node0();
	do_setaffinity(0);
	// return 0;

	config_info.sock_port_list = (char**)calloc_numa(config_info.nPeers * sizeof(char*));
	config_info.node_ip_list = (char**)calloc_numa(config_info.nPeers * sizeof(char*));

	for (int i = 0; i < config_info.nRanks; i++) {
		config_info.sock_port_list[i] = (char*)calloc_numa(8);
		config_info.node_ip_list[i] = (char*)calloc_numa(128);

		sprintf(config_info.sock_port_list[i], "%d", 13410 + i);
		sprintf(config_info.node_ip_list[i], "%s", argv[5 + i]);
	}

	ret = init_env();
	CHECK(ret == 0, "Failed to init env");

	nRanks = config_info.nRanks;
	rank = config_info.rank;
	msg_size = config_info.msg_size;

	recv_all_flag = (uint8_t*)calloc_numa(1);
	polling_flag = (uint8_t*)calloc_numa(1);
	send_all_flag = (uint8_t*)calloc_numa(1);

	*recv_all_flag = 0;
	*send_all_flag = 0;
	*polling_flag = 1;

	ret = setup_ib(config_info.nRanks);
	CHECK(ret == 0, "Failed to setup IB");

	// size_t buf_size = ib_res.ib_buf_size;
	char* send_addr = ib_res.ib_send_buf;
	char* recv_addr = ib_res.ib_recv_buf + msg_size;

	send_thread = (pthread_t*)calloc_numa(nRanks * sizeof(pthread_t));
	for (i = 0; i < nRanks; i++) {
		if (i != rank)
			pthread_create(&send_thread[i], NULL, send_thread_func, (void*)(uint64_t)i);
	}

	debug("send_addr = %" PRIx64 "", (uint64_t)send_addr);
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
		}
		recv_addr += msg_size;
	}

	memory_barrier();
	while (*send_all_flag == (uint8_t)0)
		_mm_pause();

	clock_gettime(CLOCK_MONOTONIC, &time2);
	diffInNanos = (time2.tv_sec - time1.tv_sec) * (long)1e9 + (time2.tv_nsec - time1.tv_nsec);
	log("Rank-[%d] wait send_all_flag Use time [%.3lf] us", rank, (double)diffInNanos / 1000);

	sock_barrier(nRanks, rank);

	*polling_flag = 0;
	pthread_join(ibv_polling_t, NULL);
error:
	// if (ibvhandle != NULL) dlclose(ibvhandle);
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
	sprintf(fname, "rank-[%d].log", config_info.rank);

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
