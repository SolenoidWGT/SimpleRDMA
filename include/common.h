/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef __COMMON_H__
#define __COMMON_H__

#include "nccl.h"
#include <stdint.h>
#include <stdio.h>
// #include <algorithm>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char const* ncclGetLastError(ncclComm_t comm);
#define CUDACHECK(cmd)                                                                                           \
	do {                                                                                                         \
		cudaError_t err = cmd;                                                                                   \
		if (err != cudaSuccess) {                                                                                \
			char hostname[1024];                                                                                 \
			getHostName(hostname, 1024);                                                                         \
			printf("%s: Test CUDA failure %s:%d '%s'\n", hostname, __FILE__, __LINE__, cudaGetErrorString(err)); \
			return testCudaError;                                                                                \
		}                                                                                                        \
	} while (0)

#define NCCLCHECK(cmd)                                                                             \
	do {                                                                                           \
		ncclResult_t res = cmd;                                                                    \
		if (res != ncclSuccess) {                                                                  \
			char hostname[1024];                                                                   \
			getHostName(hostname, 1024);                                                           \
			printf("%s: Test NCCL failure %s:%d "                                                  \
			       "'%s / %s'\n",                                                                  \
			       hostname, __FILE__, __LINE__, ncclGetErrorString(res), ncclGetLastError(NULL)); \
			return testNcclError;                                                                  \
		}                                                                                          \
	} while (0)

typedef enum {
	testSuccess = 0,
	testInternalError = 1,
	testCudaError = 2,
	testNcclError = 3,
	testTimeout = 4,
	testNumResults = 5
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd)                                                                             \
	do {                                                                                           \
		testResult_t r = cmd;                                                                      \
		if (r != testSuccess) {                                                                    \
			char hostname[1024];                                                                   \
			getHostName(hostname, 1024);                                                           \
			printf(" .. %s pid %d: Test failure %s:%d\n", hostname, getpid(), __FILE__, __LINE__); \
			return r;                                                                              \
		}                                                                                          \
	} while (0)

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
	gethostname(hostname, maxlen);
	for (int i = 0; i < maxlen; i++) {
		if (hostname[i] == '.') {
			hostname[i] = '\0';
			return;
		}
	}
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"


extern int test_ncclVersion; // init'd with ncclGetVersion()
extern int test_opnum;
extern int test_typenum;
extern ncclDataType_t test_types[ncclNumTypes];
extern const char* test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[];
extern const char* test_opnames[];

testResult_t ncclAlltoAll(void* sendbuff, void* recvbuff, size_t count, ncclDataType_t type, ncclComm_t comm,
                          cudaStream_t stream);
#endif
