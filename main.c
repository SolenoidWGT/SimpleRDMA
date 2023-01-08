#define _GNU_SOURCE /* See feature_test_macros(7) */

#include "bruck.h"
#include "client.h"
#include "config.h"
#include "debug.h"
#include "ib.h"
#include "numa.h"
#include "pci/pci.h"
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

bool use_pcie_relaxed_order;
size_t chunk_size; // 512 KB
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

static bool check_pcie_relaxed_ordering_compliant(void) {
	struct pci_access* pacc;
	struct pci_dev* dev;
	bool cpu_is_RO_compliant = true;

	pacc = pci_alloc();
	pci_init(pacc);
	pci_scan_bus(pacc);
	for (dev = pacc->devices; dev && cpu_is_RO_compliant; dev = dev->next) {
		pci_fill_info(dev, PCI_FILL_IDENT | PCI_FILL_BASES | PCI_FILL_CLASS);
		/* https://lore.kernel.org/patchwork/patch/820922/ */
		if ((dev->vendor_id == 0x8086) && (((dev->device_id >= 0x6f01 && dev->device_id <= 0x6f0e) ||
		                                    (dev->device_id >= 0x2f01 && dev->device_id <= 0x2f01))))
			cpu_is_RO_compliant = false;
	}
	pci_cleanup(pacc);
	return cpu_is_RO_compliant;
}

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

	// if (local_node == -1)
	// 	local_node = cal_numa_node(locak_rank, nRanks, task_per_node);

	// if (cpu == -1)
	// 	cpu = get_cpu_for_rank(locak_rank, nRanks, tid, local_node);

	for (int i = 4 * tid; i < (4 * tid + 4); i++) {
		CPU_SET(node_list[0].cpu_numa_list[i], &mask);
	}

	if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
		log_warn("warning: could not set CPU affinity, continuing...\n");
	}
	log("Rank-[%d], tid-[%d], set on Node-[%d]'s cpu-[%d]", rank, tid, local_node, cpu);
}

// struct ibv_mr* (*ibv_internal_reg_mr_iova2)(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova, int access);
int main(int argc, char* argv[]) {

	int ret = 0;
	config_info.num_concurr_msgs = 1;
	config_info.nRanks = atoi(argv[1]);
	config_info.nPeers = config_info.nRanks - 1;
	config_info.rank = atoi(argv[2]);
	config_info.msg_size = atoi(argv[3]);
	config_info.task_per_node = atoi(argv[4]);
	// config_info.base = atoi(argv[5]);
	// config_info.msg_size = 16;

	CHECK(init_env() == 0, "Failed to init env");
	bool nccl_test = true;
	bool use_chunk = false;
	int nDevs = 8;
	local_node = 0;
	use_pcie_relaxed_order = false;
	get_cpu_mask();
	do_setaffinity(MAIN_THREAD_ID, -1);

	// return 0;

	config_info.sock_port_list = (char**)calloc_numa(config_info.nPeers * sizeof(char*));
	config_info.node_ip_list = (char**)calloc_numa(config_info.nPeers * sizeof(char*));

	for (int i = 0; i < config_info.nRanks; i++) {
		config_info.sock_port_list[i] = (char*)calloc_numa(8);
		config_info.node_ip_list[i] = (char*)calloc_numa(128);

		sprintf(config_info.sock_port_list[i], "%d", 13410 + i);
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

	if (use_pcie_relaxed_order) {
		log("Support PCIe relaxed order!");
	} else {
		log("Not support PCIe relaxed order!");
	}

	/* connect QP */
	CHECK(sock_handshack(nRanks, rank) == 0, "sock_handshack  error");
	CHECK(setup_ib(config_info.nRanks) == 0, "Failed to setup IB");
	CHECK(setup_ib_buffer(nDevs) == 0, "Setip ib");

	all2AllBruck_nGPUs(16, nDevs, msg_size, (rank == 0) ? 0 : 8, rank, nRanks);

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
