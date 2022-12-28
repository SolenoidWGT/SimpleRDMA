#ifndef SETUP_IB_H_
#define SETUP_IB_H_

#include "ib.h"
#include <infiniband/verbs.h>

#ifndef likely
#define likely(x) __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

#define CPU_NUMS 128
extern int numa_cpu_num;
extern int cpu_numa_list[CPU_NUMS];
inline static void memory_barrier() { asm volatile("" ::: "memory"); }

inline static int get_cpu_for_rank(int rank, int nRanks, int pthread_id) {
	// 15 sender thread, 1 polling thread, 1 main thread.
	return cpu_numa_list[(rank * (nRanks + 2) + pthread_id) % numa_cpu_num];
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
extern pthread_t ibv_polling_t;

int setup_ib(int);
void close_ib_connection();
void* pollingFunc(void* vargs);
// int connect_qp_server();
int connect_qp_client();

#endif /*setup_ib.h*/
