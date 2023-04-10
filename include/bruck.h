#ifndef BRUCK_H
#define BRUCK_H
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#define BRUCK_TYPE float

int all2AllBruck(int rank, int nRanks, int localRank, size_t msgSize, size_t all2all_size, double nic_flow_size);
long IBSendRecvP2P(int nRanks, int rank, char* send_buff, char* recv_buff, int dst_peer, int src_peer, int msg_size);
int all2AllBruck_nGPUs(int dev_nRanks, int nDevs, size_t block_size, int base, int socket_rank, int socket_nRanks);
#endif