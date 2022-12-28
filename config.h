#ifndef CONFIG_H_
#define CONFIG_H_

#include <inttypes.h>
#include <stdbool.h>

enum ConfigFileAttr {
	ATTR_SERVERS = 1,
	ATTR_CLIENTS,
	ATTR_MSG_SIZE,
	ATTR_NUM_CONCURR_MSGS,
};

struct ConfigInfo {
	// int num_servers;
	// int num_clients;
	int nRanks;
	int nPeers;
	char** node_ip_list;
	char** sock_port_list;
	// char** servers; /* list of servers */
	// char** clients; /* list of clients */

	// bool is_server; /* if the current node is server */
	int rank; /* the rank of the node */

	int msg_size;         /* the size of each echo message */
	int num_concurr_msgs; /* the number of messages can be sent concurrently */
	int task_per_node;

} __attribute__((aligned(64)));

extern struct ConfigInfo config_info;

static inline int get_rank_index(int rank) {
	if (rank <= config_info.rank)
		return rank;
	else
		return rank - 1;
}

int parse_config_file(char* fname);
void destroy_config_info();

void print_config_info();

// extern char* SERVER_PORT;
// extern char* CLIENT_PORT;

#endif /* CONFIG_H_*/
