#include "setup_ib.h"
#include "config.h"
#include "debug.h"
#include "ib.h"
#include "numa.h"
#include "sock.h"
#include <arpa/inet.h>
#include <malloc.h>
#include <unistd.h>

#define DEBUG_IB 1

struct IBRes ib_res;
// struct MRinfo* recvMRinfo = NULL;
pthread_t sock_server_t;
pthread_mutex_t sock_mutex;

int* peer_sockfd = NULL;
int local_sockfd = 0;

struct QPInfo* local_qp_info = NULL;
struct QPInfo* remote_qp_info = NULL;

void* calloc_numa(size_t size);

void close_sock(int nRanks) {
	int i;
	if (peer_sockfd != NULL) {
		for (i = 0; i < nRanks; i++) {
			if (peer_sockfd[i] > 0) {
				close(peer_sockfd[i]);
			}
		}
		free_numa(peer_sockfd);
		peer_sockfd = NULL;
	}
	if (local_sockfd > 0) {
		close(local_sockfd);
	}
}

void* calloc_numa(size_t size) {
	void* addr = numa_alloc_onnode(size, local_node);
	memset(addr, 0, size);
	return addr;
}

void free_numa(void* addr) {
	// numa_free(addr, );
}

struct ExcMRArgs {
	struct IBMemInfo* mr_ptr;
	bool is_server;
	bool is_meta_mr;
};

void* sock_exchange_MR(void* args) {
	bool is_meta_mr = ((struct ExcMRArgs*)args)->is_meta_mr;
	bool is_server = ((struct ExcMRArgs*)args)->is_server;
	struct IBMemInfo* mr_ptr = ((struct ExcMRArgs*)args)->mr_ptr;
	struct MRinfo* recv_mr;
	int mr_id = mr_ptr->mr_id;
	int ret = 0, i = 0;
	int nRanks = config_info.nRanks;
	int rank = config_info.rank;
	int connect_index = nRanks - rank;

	int start = is_server ? connect_index : 0;
	int end = is_server ? nRanks : connect_index;
	const char* info_str = is_server ? "ServerSide" : "ClientSide";

	CHECK(remote_qp_info != NULL, "remote_qp_info not init");
	CHECK(local_qp_info != NULL, "local_qp_info not init");
	CHECK(peer_sockfd != NULL, "peer_sockfd not init");

	/* +++++++++++++++++++++++++++ MR exchange start +++++++++++++++++++++++++++ */
	/* send mr_info to client */
	for (i = start; i < end; i++) {
		if (i != rank) {
			if (mr_ptr->mr_info.addr == NULL || mr_ptr->mr_info.length == 0 || mr_ptr->mr_info.rkey == 0) {
				log("ERROR! \"%s\" sock_set_mr_info NULL", info_str);
				// goto error;
			}

			ret = sock_set_mr_info(peer_sockfd[i], &(mr_ptr->mr_info));
			CHECK(ret == 0, "Failed to send mr_info to client[%d]", i);
		}
	}

	for (i = start; i < end; i++) {
		if (i != rank) {
			if (is_meta_mr)
				recv_mr = &(ib_res.remote_meta_recv_mr_info[i]);
			else
				recv_mr = &(ib_res.remote_mr_info[i][mr_id]);

			ret = -1;
			// while (ret != 0) {
			ret = sock_get_MR_info(peer_sockfd[i], recv_mr);
			// usleep(1000);
			//}
			// CHECK(ret == 0, "Failed to get mr_info[%d] from server", i);

			if (recv_mr->addr == NULL || recv_mr->length == 0 || recv_mr->rkey == 0) {
				log("ERROR! \"%s\" sock_get_MR_info NULL", info_str);
			}
#ifdef DEBUG_IB
			log("\"%s\": Remote Rank-[%d] mr info addr:%p, length:%ld, rkey: %u", info_str, i, recv_mr->addr,
			    recv_mr->length, recv_mr->rkey);
#endif
		}
	}
	/* +++++++++++++++++++++++++++ MR exchange end +++++++++++++++++++++++++++ */
	return (void*)0;

error:
	return (void*)-1;
}

void* sock_exchange_QP(void* args) {
	bool is_server = (bool)(int64_t)args;
	int ret = 0, i = 0;
	int nRanks = config_info.nRanks;
	int rank = config_info.rank;
	int connect_index = nRanks - rank;

	int start = is_server ? connect_index : 0;
	int end = is_server ? nRanks : connect_index;

	CHECK(remote_qp_info != NULL, "remote_qp_info not init");
	CHECK(local_qp_info != NULL, "local_qp_info not init");
	CHECK(peer_sockfd != NULL, "peer_sockfd not init");

	for (i = start; i < end; i++) {
		if (i != rank) {
			local_qp_info[i].lid = ib_res.port_attr.lid;
			local_qp_info[i].qp_num = ib_res.qp[i]->qp_num;
			local_qp_info[i].rank = config_info.rank;
		}
	}

	/* send qp_info to client */
	// (WGT): Send must ahead befor Recv ???
	for (i = start; i < end; i++) {
		if (i != rank) {
			ret = sock_set_qp_info(peer_sockfd[i], &local_qp_info[i]);
			CHECK(ret == 0, "Failed to send qp_info to client[%d]", i);
		}
	}

	/* get qp_info from client */
	for (i = start; i < end; i++) {
		if (i != rank) {
			ret = sock_get_qp_info(peer_sockfd[i], &remote_qp_info[i]);
			CHECK(ret == 0, "Failed to get qp_info from client[%d]", i);
		}
	}

	/* change send QP state to RTS */
	// log(LOG_SUB_HEADER, "Start of IB Config");
	for (i = start; i < end; i++) {
		if (i != rank) {
			ret = modify_qp_to_rts(ib_res.qp[i], remote_qp_info[i].qp_num, remote_qp_info[i].lid);
			CHECK(ret == 0, "Failed to modify qp[%d] to rts", i);
#ifdef DEBUG_IB
			const char* info_str = is_server ? "ServerSide" : "ClientSide";
			log("\t \"%s\": Rank-[%d]:qp[%" PRIu32 "] <-> Rank-[%d]:qp[%" PRIu32 "]", info_str, rank,
			    ib_res.qp[i]->qp_num, i, remote_qp_info[i].qp_num);
#endif
		}
	}
	return (void*)0;

error:
	return (void*)-1;
}

void* socket_accept_thread(void* arg) {
	int i, idx = (int)(uint64_t)arg;
	for (i = 0; i < idx; i++) {
		if (i != config_info.rank) {
			// pthread_mutex_lock(&sock_mutex);
			if (peer_sockfd[i] == 0) {
				peer_sockfd[i] = sock_create_connect(config_info.node_ip_list[i], config_info.sock_port_list[i]);
				CHECK(peer_sockfd[i] > 0, "Failed to create peer_sockfd[%d]", i);
			}
			// pthread_mutex_unlock(&sock_mutex);
		}
	}
	return (void*)0;
error:
	return (void*)-1;
}

int sock_handshack(int nRanks, int rank) {
	int i, ret;
	int idx = rank;
	struct sockaddr_in peer_addr;
	pthread_t sock_conn_t;

	socklen_t peer_addr_len = sizeof(struct sockaddr_in);

	peer_sockfd = (int*)calloc_numa(nRanks * sizeof(int));
	CHECK(peer_sockfd != NULL, "Failed to allocate peer_sockfd");

	local_sockfd = sock_create_bind(config_info.sock_port_list[rank]);
	CHECK(local_sockfd > 0, "Failed to create server socket.");
	listen(local_sockfd, 5);

	ret = pthread_create(&sock_conn_t, NULL, socket_accept_thread, (void*)(int64_t)idx);
	CHECK(ret == 0, "socket_accept_thread thread create error");

	for (i = idx; i < nRanks; i++) {
		if (i != rank) {
			// pthread_mutex_lock(&sock_mutex);
			if (peer_sockfd[i] == 0) {
				peer_sockfd[i] = accept(local_sockfd, (struct sockaddr*)&peer_addr, &peer_addr_len);
				CHECK(peer_sockfd[i] > 0, "Failed to create peer_sockfd[%d]", i);
			}
			// pthread_mutex_unlock(&sock_mutex);
		}
	}

	pthread_join(sock_conn_t, NULL);
	log("sock_handshack success!");
	return 0;
error:
	close_sock(nRanks);
	return -1;
}

int register_ib_mr(void* buffer, size_t size, struct ibv_mr** mr, struct MRinfo* mrInfo) {
	/* set the buf_size twice as large as msg_size * num_concurr_msgs */
	/* the recv buffer occupies the first half while the sending buffer */
	/* occupies the second half */
	/* assume all msgs are of the same content */
	int flag = (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
	CHECK(buffer != NULL, "Failed to allocate mr.addr");
	memset(buffer, 0, size);

	if (use_pcie_relaxed_order) {
		// https://lore.kernel.org/linux-rdma/20191119192919.GA16030@ziepe.ca/T/
		// The new function would check against the kernel whether relaxed ordering is supported or not, and disable it
		// if necessary.
		flag |= IBV_ACCESS_RELAXED_ORDERING;
		*mr = ibv_reg_mr_iova2(ib_res.pd, (void*)buffer, size, (uint64_t)buffer, flag);
	} else {
		*mr = ibv_reg_mr(ib_res.pd, (void*)buffer, size, flag);
	}
	CHECK(*mr != NULL, "Failed to register mr");

	if (mrInfo != NULL) {
		mrInfo->addr = (*mr)->addr;
		mrInfo->length = (*mr)->length;
		mrInfo->rkey = (*mr)->rkey;
		log("Local mr info addr:%p, length:%ld, rkey: %u", mrInfo->addr, mrInfo->length, mrInfo->rkey);
	}

	return 0;
error:
	return -1;
}

int exchange_mr_info(struct IBMemInfo* mr_ptr, bool is_meta) {
	struct ExcMRArgs arg1, arg2;
	arg1.mr_ptr = mr_ptr;
	arg1.is_server = true;
	arg1.is_meta_mr = is_meta;
	CHECK(pthread_create(&sock_server_t, NULL, sock_exchange_MR, (void*)(&arg1)) == 0,
	      "sock_exchange_MR thread create error");

	arg2.mr_ptr = mr_ptr;
	arg2.is_server = false;
	arg2.is_meta_mr = is_meta;
	sock_exchange_MR((void*)(&arg2));

	pthread_join(sock_server_t, NULL);
	return 0;
error:
	return -1;
}

void exchange_qp_info() {
	void* re;
	int ret = pthread_create(&sock_server_t, NULL, sock_exchange_QP, (void*)(int64_t) true);
	CHECK(ret == 0, "sock_exchange_QP thread create error");
	ret = (int)(uint64_t)sock_exchange_QP(false);
	pthread_join(sock_server_t, &re);
	CHECK((int)(uint64_t)re == 0 && ret == 0, "Failed to sock_exchange_QP");
error:
	return;
}

int setup_ib_buffer(int mr_nums, int nDevs) {
	ib_res.send_mr_info = (struct IBMemInfo*)calloc_numa(mr_nums * sizeof(struct IBMemInfo));
	ib_res.recv_mr_info = (struct IBMemInfo*)calloc_numa(mr_nums * sizeof(struct IBMemInfo));
	ib_res.remote_meta_recv_mr_info = (struct MRinfo*)calloc_numa(mr_nums * sizeof(struct MRinfo));
	ib_res.remote_mr_info = (struct MRinfo**)calloc_numa(mr_nums * sizeof(struct MRinfo*));

	for (int i = 0; i < nDevs; i++) {
		ib_res.remote_mr_info[i] = (struct MRinfo*)calloc_numa(sizeof(struct MRinfo));
	}
	ib_res.mr_nums = mr_nums;
	// ib_res.remote_mr_nums = 0;
	ib_res.wc = (struct ibv_wc*)calloc_numa(MAX_WC_NUMS * sizeof(struct ibv_wc));
	return 0;
}

int alloc_ib_buffer(int nRanks, struct IBMemInfo* buff, bool is_meta, size_t buff_size) {
	/* init remote mr */
	CHECK(buff != NULL, "Failed to allocate remote mr info");
	buff->mr_id = 0;

	/* register mr */
	// 	if (numa_available() == 0 && numa_num_configured_nodes() > 1) {
	// #ifdef DEBUG_IB
	// 		log("NUMA is Enable!");
	// #endif
	// 		buff->addr = (char*)numa_alloc_onnode(buff_size, local_node);
	// 	} else {
	buff->addr = (char*)memalign(4096, buff_size);
	//}

	CHECK(register_ib_mr(buff->addr, buff_size, &(buff->mr), &(buff->mr_info)) == 0, "mr");
	CHECK(exchange_mr_info(buff, is_meta) == 0, "mr exchange");

	log("IB setup MR success!");
	return 0;
error:
	return -1;
}

int setup_ib(int nRanks) {
	int ret = 0;
	int i = 0;
	int num_ib_cards;
	int rank = config_info.rank;
	struct ibv_device** dev_list = NULL;
	memset(&ib_res, 0, sizeof(struct IBRes));

	ib_res.num_qps = nRanks;

	/* init qp info */
	remote_qp_info = (struct QPInfo*)calloc_numa(nRanks * sizeof(struct QPInfo));
	CHECK(remote_qp_info != NULL, "Failed to allocate remote_qp_info");

	local_qp_info = (struct QPInfo*)calloc_numa(nRanks * sizeof(struct QPInfo));
	CHECK(local_qp_info != NULL, "Failed to allocate local_qp_info");

	/* get IB device list */
	dev_list = ibv_get_device_list(&num_ib_cards);
	CHECK(dev_list != NULL, "Failed to get ib device list.");

	for (i = 0; i < num_ib_cards; i++) {
		/* create IB context */
		ib_res.ctx = ibv_open_device(dev_list[i]);
		CHECK(ib_res.ctx != NULL, "Failed to open ib device.");

		/* query IB port attribute */
		ret = ibv_query_port(ib_res.ctx, IB_PORT, &ib_res.port_attr);
		CHECK(ret == 0, "Failed to query IB port information.");

		if (ib_res.port_attr.state != IBV_PORT_ACTIVE || (ib_res.port_attr.sm_lid == 0 && ib_res.port_attr.lid == 0)) {
			ret = ibv_close_device(ib_res.ctx);
			CHECK(ret == 0, "Failed to close ib device.");
			continue;
		}

		/* query IB device attr */
		ret = ibv_query_device(ib_res.ctx, &ib_res.dev_attr);
		CHECK(ret == 0, "Failed to query device");

		/* allocate protection domain */
		ib_res.pd = ibv_alloc_pd(ib_res.ctx);
		CHECK(ib_res.pd != NULL, "Failed to allocate protection domain.");

		break;
	}

	/* create cq */
#ifdef DEBUG_IB
	log("ib_res.dev_attr.max_cqe is %d", ib_res.dev_attr.max_cqe);
#endif
	ib_res.cq = ibv_create_cq(ib_res.ctx, 2048, NULL, NULL, 0);
	CHECK(ib_res.cq != NULL, "Failed to create cq");

	/* create srq */
	struct ibv_srq_init_attr srq_init_attr = {
	    .attr.max_wr = ib_res.dev_attr.max_srq_wr,
	    .attr.max_sge = 1,
	};

	ib_res.srq = ibv_create_srq(ib_res.pd, &srq_init_attr);

	/* create qp */
	struct ibv_qp_init_attr qp_init_attr;
	memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
	qp_init_attr.qp_type = IBV_QPT_RC;
	qp_init_attr.send_cq = ib_res.cq;
	qp_init_attr.recv_cq = ib_res.cq;
	qp_init_attr.srq = ib_res.srq;
	qp_init_attr.cap.max_send_wr = MAX_WC_NUMS;
	qp_init_attr.cap.max_recv_wr = MAX_WC_NUMS;
	qp_init_attr.cap.max_send_sge = 4;
	qp_init_attr.cap.max_recv_sge = 1;
	qp_init_attr.cap.max_inline_data = 60;
	// qp_init_attr.xrc_domain = NULL;
	qp_init_attr.sq_sig_all = 0;

	ib_res.qp = (struct ibv_qp**)calloc_numa(ib_res.num_qps * sizeof(struct ibv_qp*));
	CHECK(ib_res.qp != NULL, "Failed to allocate qp");

#ifdef DEBUG_IB
	log("num qpes %d", ib_res.num_qps);
#endif

	for (i = 0; i < ib_res.num_qps; i++) {
		if (i != rank) {
			ib_res.qp[i] = ibv_create_qp(ib_res.pd, &qp_init_attr);
			CHECK(ib_res.qp[i] != NULL, "Failed to create qp[%d]", i);
		} else
			ib_res.qp[i] = NULL;
	}

	exchange_qp_info();

	/*
	CHECK(alloc_ib_buffer(nRanks, &(ib_res.meta_recv_mr_info), true, nRanks * 1024 * 1024) == 0, "alloc ib buffer");
	char* buf_ptr = ib_res.meta_recv_mr_info.addr;
	char* buf_base = buf_ptr;
	size_t buf_offset = 0;
	size_t buff_size = ib_res.meta_recv_mr_info.mr->length;
	size_t msg_size = buff_size / nRanks;
	uint32_t lkey = ib_res.meta_recv_mr_info.mr->lkey;
	for (int i = 0; i < nRanks; i++) {
	    for (int j = 0; j < config_info.num_concurr_msgs; j++) {
	        CHECK(post_srq_recv(msg_size, lkey, (uint64_t)buf_ptr, ib_res.srq, buf_ptr) == 0, "post_srq_recv");
	        buf_offset = (buf_offset + msg_size) % buff_size;
	        buf_ptr = buf_base + buf_offset;
	    }
	}
	*/
	sock_barrier(nRanks, rank);
	log("IB setup QP success!");

	ibv_free_device_list(dev_list);

	return 0;

error:
	close_sock(nRanks);
	if (dev_list != NULL) {
		ibv_free_device_list(dev_list);
	}
	return -1;
}

void close_ib_connection() {
	int i;

	if (ib_res.qp != NULL) {
		for (i = 0; i < ib_res.num_qps; i++) {
			if (ib_res.qp[i] != NULL) {
				ibv_destroy_qp(ib_res.qp[i]);
			}
		}
		free_numa(ib_res.qp);
	}

	if (ib_res.srq != NULL) {
		ibv_destroy_srq(ib_res.srq);
	}

	if (ib_res.cq != NULL) {
		ibv_destroy_cq(ib_res.cq);
	}

	for (int i = 0; i < ib_res.mr_nums; i++) {
		if (ib_res.send_mr_info[i].mr != NULL) {
			ibv_dereg_mr(ib_res.send_mr_info[i].mr);
		}

		if (ib_res.recv_mr_info[i].mr != NULL) {
			ibv_dereg_mr(ib_res.recv_mr_info[i].mr);
		}
		if (ib_res.recv_mr_info[i].addr != NULL) {
			if (numa_available() == 0 && numa_num_configured_nodes() > 1) {
				numa_free(ib_res.recv_mr_info[i].addr, ib_res.recv_mr_info[i].mr_info.length);
				numa_free(ib_res.send_mr_info[i].addr, ib_res.send_mr_info[i].mr_info.length);
			} else {
				free_numa(ib_res.recv_mr_info[i].addr);
				free_numa(ib_res.send_mr_info[i].addr);
			}
		}
	}

	if (ib_res.pd != NULL) {
		ibv_dealloc_pd(ib_res.pd);
	}

	if (ib_res.ctx != NULL) {
		ibv_close_device(ib_res.ctx);
	}
}
