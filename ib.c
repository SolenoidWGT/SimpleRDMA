#include "ib.h"
#include "debug.h"
#include <arpa/inet.h>
#include <unistd.h>
int modify_qp_to_rts(struct ibv_qp* qp, uint32_t target_qp_num, uint16_t target_lid) {
	int ret = 0;

	/* change QP state to INIT */
	{
		struct ibv_qp_attr qp_attr;
		memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
		qp_attr.qp_state = IBV_QPS_INIT, qp_attr.pkey_index = 0, qp_attr.port_num = IB_PORT,
		qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
		// , IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS
		ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
		CHECK(ret == 0, "Failed to modify qp to INIT.");
	}

	/* Change QP state to RTR */
	{
		struct ibv_qp_attr qp_attr;
		memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
		qp_attr.qp_state = IBV_QPS_RTR;
		qp_attr.path_mtu = IB_MTU;
		qp_attr.dest_qp_num = target_qp_num;
		qp_attr.rq_psn = 0;
		qp_attr.max_dest_rd_atomic = 1;
		qp_attr.min_rnr_timer = 12;
		qp_attr.ah_attr.is_global = 0;
		qp_attr.ah_attr.dlid = target_lid;
		qp_attr.ah_attr.sl = IB_SL;
		qp_attr.ah_attr.src_path_bits = 0;
		qp_attr.ah_attr.port_num = IB_PORT;
		ret = ibv_modify_qp(qp, &qp_attr,
		                    IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
		                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
		CHECK(ret == 0, "Failed to change qp to rtr.");
	}

	/* Change QP state to RTS */
	{
		struct ibv_qp_attr qp_attr;
		memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
		qp_attr.qp_state = IBV_QPS_RTS;
		// qp_attr.cur_qp_state = IBV_QPS_RESET;
		qp_attr.timeout = 14;
		qp_attr.retry_cnt = 7;
		qp_attr.rnr_retry = 12;
		qp_attr.sq_psn = 0;
		qp_attr.max_rd_atomic = 1;
		ret = ibv_modify_qp(qp, &qp_attr,
		                    IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
		                        IBV_QP_MAX_QP_RD_ATOMIC);
		CHECK(ret == 0, "Failed to modify qp to RTS.");
	}

	return 0;
error:
	return -1;
}

int post_send(uint32_t req_size, uint32_t lkey, uint64_t wr_id, uint32_t imm_data, struct ibv_qp* qp, char* buf) {
	int ret = 0;
	struct ibv_send_wr* bad_send_wr;

	struct ibv_sge list = {.addr = (uintptr_t)buf, .length = req_size, .lkey = lkey};

	struct ibv_send_wr send_wr = {.wr_id = wr_id,
	                              .sg_list = &list,
	                              .num_sge = 1,
	                              .opcode = IBV_WR_SEND_WITH_IMM,
	                              .send_flags = IBV_SEND_SIGNALED,
	                              .imm_data = htonl(imm_data)};

	ret = ibv_post_send(qp, &send_wr, &bad_send_wr);
	return ret;
}

int post_srq_recv(uint32_t req_size, uint32_t lkey, uint64_t wr_id, struct ibv_srq* srq, char* buf) {
	int ret = 0;
	struct ibv_recv_wr* bad_recv_wr;

	struct ibv_sge list = {.addr = (uintptr_t)buf, .length = req_size, .lkey = lkey};

	struct ibv_recv_wr recv_wr = {.wr_id = wr_id, .sg_list = &list, .num_sge = 1};

	ret = ibv_post_srq_recv(srq, &recv_wr, &bad_recv_wr);
	return ret;
}

// uint32_t req_size, uint32_t lkey, uint64_t wr_id, uint32_t imm_data, struct ibv_qp *qp, char *buf

int post_write_imm(uint32_t req_size, uint32_t lkey, uint64_t wr_id, struct ibv_qp* qp, char* buf, uint32_t rkey,
                   void* remote_addr, uint32_t imm_data) {
	struct ibv_send_wr send_wr, *bad_wr = NULL;
	struct ibv_sge sge;
	int err = 0;
	memset(&send_wr, 0, sizeof(struct ibv_send_wr));

	send_wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
	send_wr.imm_data = htonl(imm_data);

	send_wr.wr_id = (uintptr_t)wr_id;
	send_wr.sg_list = &sge;
	send_wr.num_sge = 1;
	send_wr.send_flags = IBV_SEND_SIGNALED;
	send_wr.wr.rdma.remote_addr = (uintptr_t)remote_addr; // WGT
	send_wr.wr.rdma.rkey = rkey;

	sge.addr = (uintptr_t)buf;
	sge.length = req_size;
	sge.lkey = lkey;

	err = ibv_post_send(qp, &send_wr, &bad_wr);
	return err;
}

int post_write(uint32_t req_size, uint32_t lkey, uint64_t wr_id, struct ibv_qp* qp, char* buf, uint32_t rkey,
               void* remote_addr) {
	struct ibv_send_wr send_wr, *bad_wr = NULL;
	struct ibv_sge sge;
	int err = 0;
	memset(&send_wr, 0, sizeof(struct ibv_send_wr));

	send_wr.opcode = IBV_WR_RDMA_WRITE;

	send_wr.wr_id = (uintptr_t)wr_id;
	send_wr.sg_list = &sge;
	send_wr.num_sge = 1;
	send_wr.send_flags = IBV_SEND_SIGNALED;
	send_wr.wr.rdma.remote_addr = (uintptr_t)remote_addr; // WGT
	send_wr.wr.rdma.rkey = rkey;

	sge.addr = (uintptr_t)buf;
	sge.length = req_size;
	sge.lkey = lkey;

	err = ibv_post_send(qp, &send_wr, &bad_wr);
	return err;
}
