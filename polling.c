#define _GNU_SOURCE
#include "client.h"
#include "config.h"
#include "debug.h"
#include "ib.h"
#include "setup_ib.h"
#include <stdbool.h>
#include <stdlib.h>
#include <sys/time.h>

pthread_t polling_thread;
volatile uint8_t* polling_flag;
volatile uint8_t* recv_all_flag;
volatile uint8_t* send_all_flag;

struct pollingArgs {
	int num_wc;
	int thread_id;
	size_t msg_size;
};

int createPollingThread() {
	struct pollingArgs* args = (struct pollingArgs*)calloc_numa(sizeof(struct pollingArgs));
	int retval = pthread_create(&polling_thread, NULL, pollingFunc, (void*)args);
	if (retval) {
		log("pthread create error.");
		return -1;
	}
	return 0;
}

void* pollingFunc(void* vargs) {
	int n, i, ret, ops_count = 0;
	// struct pollingArgs* args = (struct pollingArgs*)vargs;
	// int num_wc = args->num_wc;
	// int thread_id = args->thread_id;
	// ib_res is gloabl var
	// struct ibv_qp **qp = ib_res.qp;
	int64_t thread_id = 1;
	int num_wc = 15;
	int num_acked_peers = 0;
	int success_op_nums = 0;
	struct ibv_cq* cq = ib_res.cq;
	struct ibv_srq* srq = ib_res.srq;
	struct ibv_wc* wc = NULL;

	// buffer info
	// size_t buf_size = ib_res.ib_buf_size;
	uint32_t lkey = ib_res.recv_mr_info[0].mr->lkey;
	// char* buf_ptr = ib_res.recv_mr_info.addr ;
	// char* buf_base = ib_res.recv_mr_info.addr;
	// int buf_offset = 0;

	// rank info
	int nRanks = config_info.nRanks;
	// int rank = config_info.rank;
	int nPeers = nRanks - 1;
	// int num_concurr_msgs = config_info.num_concurr_msgs;
	size_t msg_size = config_info.msg_size;

	do_setaffinity(POLLING_THREAD_ID, -1);

	// malloc cq.
	wc = (struct ibv_wc*)calloc_numa(num_wc * sizeof(struct ibv_wc));
	CHECK(wc != NULL, "thread[%ld]: failed to allocate wc.", thread_id);

	/* wait for start signal */
	while (*polling_flag == 1) {

		do {
			n = ibv_poll_cq(cq, num_wc, wc);
		} while (n < 1);
		CHECK(n > 0, "pollingFunc thread[%ld]: failed to poll cq", thread_id);

		for (i = 0; i < n; i++) {
			if (wc[i].status != IBV_WC_SUCCESS) {
				if (wc[i].opcode == IBV_WC_SEND) {
					CHECK(0, "pollingFunc thread[%ld]: send failed status: %s; wr_id = %" PRIx64 "", thread_id,
					      ibv_wc_status_str(wc[i].status), wc[i].wr_id);
				} else {
					CHECK(0, "pollingFunc thread[%ld]: recv failed status: %s; wr_id = %" PRIx64 "", thread_id,
					      ibv_wc_status_str(wc[i].status), wc[i].wr_id);
				}
			}

			if (wc[i].opcode == IBV_WC_RECV) {
				ops_count += 1;
				// uint32_t imm_data = ntohl(wc[i].imm_data);
				char* msg_ptr = (char*)wc[i].wr_id;
				// debug("IBV_WC_RECV ops_count = %d, imm_data = %d", ops_count, imm_data);

				/* post a new receive */
				ret = post_srq_recv(msg_size, lkey, wc[i].wr_id, srq, msg_ptr);
				CHECK(ret == 0, "pollingFunc thread[%ld]: failed to signal the client to stop", thread_id);
			} else if (wc[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
				ops_count += 1;
				uint32_t imm_data = ntohl(wc[i].imm_data);
				// char* msg_ptr = (char*)wc[i].wr_id;

				// log("IBV_WC_RECV_RDMA_WITH_IMM ops_count = %d, imm_data = %d", ops_count, imm_data);
				if (imm_data == MSG_CTL_STOP) {
					num_acked_peers += 1;
					if (num_acked_peers == nPeers) {
						memory_barrier();
						*recv_all_flag = 1;
					}
				}

				/* echo the message back */
				// post_send(msg_size, lkey, 0, imm_data, qp[imm_data], msg_ptr);
				/* post a new receive */
				// ret = post_srq_recv(msg_size, lkey, wc[i].wr_id, srq, msg_ptr);
				// CHECK(ret == 0, "pollingFunc thread[%ld]: failed to signal the client to stop", thread_id);
			} else if (wc[i].opcode == IBV_WC_RDMA_WRITE) {
				// uint32_t imm_data = ntohl(wc[i].imm_data);

				// if (imm_data == MSG_CTL_STOP) {
				success_op_nums += 1;
				if (success_op_nums == nPeers) {
					memory_barrier();
					*send_all_flag = (uint8_t)1;
				}
				//}
				// debug("IBV_WC_RDMA_WRITE[%ld] sccess!", wc[i].wr_id);

			} else if (wc[i].opcode == IBV_WC_SEND)
				debug("IBV_WC_SEND[%ld] sccess!", wc[i].wr_id);
			else
				debug("error ! unkown opcode %d", wc[i].opcode);
		} /* loop through all wc */

		if (*send_all_flag == 1 && *recv_all_flag == 1) {
			log("Send and Recv all RDMA write, ready to exit!");
			goto finish;
		}
	}
finish:
	return (void*)(int64_t)0;
error:
	// free_numa(args);
	return (void*)(int64_t)-1;
}