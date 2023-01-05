#ifndef SETUP_IB_H_
#define SETUP_IB_H_

#include "ib.h"
#include <infiniband/verbs.h>

#define CPU_NUMS 128
#define SOCKET_NUMS 4
#define USE_MULIT_THREAD 0

#define MAIN_THREAD_ID 0
#define POLLING_THREAD_ID 1
#define SENDING_THREAD_ID 2

#define WARMUP 20
#define REPEAT 300

extern int local_node;
extern int node_num;
struct cpuInfo {
	int numa_cpu_num;
	int cpu_numa_list[CPU_NUMS];
};

extern struct cpuInfo node_list[SOCKET_NUMS];
extern int cpu_numa_list[CPU_NUMS];
inline static void memory_barrier() { asm volatile("" ::: "memory"); }

inline static int get_cpu_for_rank(int local_rank, int nRanks, int pthread_id, int node) {
	// 15 sender thread, 1 polling thread, 1 main thread.
	return node_list[node].cpu_numa_list[(local_rank * (USE_MULIT_THREAD ? (nRanks - 1 + 2) : 1) + pthread_id) %
	                                     node_list[node].numa_cpu_num];
}

struct IBRes {
	struct ibv_context* ctx;
	struct ibv_pd* pd;
	struct ibv_mr* send_mr;
	struct ibv_mr* recv_mr;
	struct ibv_cq* cq;
	struct ibv_qp** qp;
	struct ibv_srq* srq;
	struct ibv_port_attr port_attr;
	struct ibv_device_attr dev_attr;

	struct MRinfo* remote_mr_info;
	struct MRinfo local_recv_mr_info;
	// struct MRinfo local_send_mr_info;

	int num_qps;
	char* ib_recv_buf;
	char* ib_send_buf;
	size_t ib_buf_size;
};

extern struct IBRes ib_res;
volatile extern uint8_t* recv_all_flag;
volatile extern uint8_t* send_all_flag;
volatile extern uint8_t* polling_flag;
// extern pthread_t ibv_polling_t;

int setup_ib(int);
void close_ib_connection();
void* pollingFunc(void* vargs);
// int connect_qp_server();
int connect_qp_client();
void do_setaffinity(int tid, int cpu);
void* calloc_numa(size_t size);
void free_numa(void* addr);
void close_sock(int nRanks);
int cal_numa_node(int locak_rank, int nRanks, int task_per_node);
int sock_handshack(int nRanks, int rank);
#endif /*setup_ib.h*/
