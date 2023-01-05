#define _GNU_SOURCE
#include "sock.h"
#include "debug.h"
#include "ib.h"
#include <arpa/inet.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

ssize_t sock_read(int sock_fd, void* buffer, size_t len) {
	ssize_t nr, tot_read;
	char* buf = buffer; // avoid pointer arithmetic on void pointer
	tot_read = 0;

	while (len != 0 && (nr = read(sock_fd, buf, len)) != 0) {
		if (nr < 0) {
			if (errno == EINTR) {
				continue;
			} else {
				return -1;
			}
		}
		len -= nr;
		buf += nr;
		tot_read += nr;
	}

	return tot_read;
}

ssize_t sock_write(int sock_fd, void* buffer, size_t len) {
	ssize_t nw, tot_written;
	const char* buf = buffer; // avoid pointer arithmetic on void pointer

	for (tot_written = 0; tot_written < len;) {
		nw = write(sock_fd, buf, len - tot_written);

		if (nw <= 0) {
			if (nw == -1 && errno == EINTR) {
				continue;
			} else {
				return -1;
			}
		}

		tot_written += nw;
		buf += nw;
	}
	return tot_written;
}

int sock_create_bind(char* port) {
	struct addrinfo hints;
	struct addrinfo *result, *rp;
	int sock_fd = -1, ret = 0;
	int option = 1;

	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_family = AF_UNSPEC;
	hints.ai_flags = AI_PASSIVE;

	ret = getaddrinfo(NULL, port, &hints, &result);
	CHECK(ret == 0, "getaddrinfo error.");

	for (rp = result; rp != NULL; rp = rp->ai_next) {
		sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));

		if (sock_fd < 0) {
			continue;
		}

		ret = bind(sock_fd, rp->ai_addr, rp->ai_addrlen);
		if (ret == 0) {
			/* bind success */
			break;
		}

		close(sock_fd);
		sock_fd = -1;
	}

	CHECK(rp != NULL, "creating socket.");

	freeaddrinfo(result);
	return sock_fd;

error:
	if (result) {
		freeaddrinfo(result);
	}
	if (sock_fd > 0) {
		close(sock_fd);
	}
	return -1;
}

int sock_create_connect(char* server_name, char* port) {
	struct addrinfo hints;
	struct addrinfo *result, *rp;
	int sock_fd = -1, ret = 0;
	int option = 1;

	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_family = AF_UNSPEC;

	ret = getaddrinfo(server_name, port, &hints, &result);
	CHECK(ret == 0, "[ERROR] %s", gai_strerror(ret));

LLOP:
	for (rp = result; rp != NULL; rp = rp->ai_next) {
		sock_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
		if (sock_fd == -1) {
			continue;
		}

		setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &option, sizeof(option));

		ret = connect(sock_fd, rp->ai_addr, rp->ai_addrlen);
		if (ret == 0) {
			/* connection success */
			break;
		}

		close(sock_fd);
		sock_fd = -1;
	}

	if (rp == NULL) {
		usleep(1000);
		goto LLOP;
	}

	// CHECK(rp != NULL, "could not connect.");
	// log("connect to %s success!", server_name);

	freeaddrinfo(result);
	return sock_fd;

error:
	if (result) {
		freeaddrinfo(result);
	}
	if (sock_fd != -1) {
		close(sock_fd);
	}
	return -1;
}

int sock_set_qp_info(int sock_fd, struct QPInfo* qp_info) {
	int n;
	struct QPInfo tmp_qp_info;

	tmp_qp_info.lid = htons(qp_info->lid);
	tmp_qp_info.qp_num = htonl(qp_info->qp_num);
	tmp_qp_info.rank = htonl(qp_info->rank);

	n = sock_write(sock_fd, (char*)&tmp_qp_info, sizeof(struct QPInfo));
	CHECK(n == sizeof(struct QPInfo), "write qp_info to socket.");

	return 0;

error:
	return -1;
}

int sock_set_mr_info(int sock_fd, struct MRinfo* mr_info) {
	int n;
	struct MRinfo tmp_mr_info;

	// Because we are all testing on x86 machines, the endianness requirements
	// are relaxed here.
	tmp_mr_info.addr = mr_info->addr;
	tmp_mr_info.length = mr_info->length;
	tmp_mr_info.rkey = mr_info->rkey;

	n = sock_write(sock_fd, (char*)&tmp_mr_info, sizeof(struct MRinfo));
	CHECK(n == sizeof(struct MRinfo), "write qp_info to socket.");

	return 0;

error:
	return -1;
}

int sock_get_qp_info(int sock_fd, struct QPInfo* qp_info) {
	int n;
	struct QPInfo tmp_qp_info;

	n = sock_read(sock_fd, (char*)&tmp_qp_info, sizeof(struct QPInfo));
	CHECK(n == sizeof(struct QPInfo), "read qp_info from socket.");

	qp_info->lid = ntohs(tmp_qp_info.lid);
	qp_info->qp_num = ntohl(tmp_qp_info.qp_num);
	qp_info->rank = ntohl(tmp_qp_info.rank);

	return 0;

error:
	return -1;
}

int sock_get_MR_info(int sock_fd, struct MRinfo* mr_info) {
	int n;
	struct MRinfo tmp_mr_info;

	n = sock_read(sock_fd, (char*)&tmp_mr_info, sizeof(struct MRinfo));
	CHECK(n == sizeof(struct MRinfo), "read mr_info .expected:[%ld]B, but get [%d]B", sizeof(struct MRinfo), n);

	mr_info->addr = tmp_mr_info.addr;
	mr_info->length = tmp_mr_info.length;
	mr_info->rkey = tmp_mr_info.rkey;

	if (mr_info->addr == NULL || mr_info->length == 0 || mr_info->rkey == 0) {
		log("ERROR! sock_get_MR_info NULL");
		goto error;
	}

	// CHECK(mr_info->addr != NULL && mr_info->length != 0 && mr_info->rkey != 0, "sock_set_mr_info get NUll mr!.");

	return 0;

error:
	return -1;
}

int sock_barrier(int nRanks, int rank) {
	int n, i;
	char ack = 1;
	CHECK(peer_sockfd != NULL, "sock_barrier.");
	for (i = 0; i < nRanks; i++) {
		if (i != rank) {
			n = sock_write(peer_sockfd[i], (char*)&ack, sizeof(char));
			CHECK(n == sizeof(char), "sock_barrier.");
			n = sock_read(peer_sockfd[i], (char*)&ack, sizeof(char));
			CHECK(n == sizeof(char), "sock_barrier.");
		}
	}

	return 0;

error:
	return -1;
}