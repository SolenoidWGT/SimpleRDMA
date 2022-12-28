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

static size_t wordSize(ncclDataType_t type) {
	switch (type) {
	case ncclChar:
#if NCCL_MAJOR >= 2
	// case ncclInt8:
	case ncclUint8:
#endif
		return 1;
	case ncclHalf:
#if defined(__CUDA_BF16_TYPES_EXIST__)
	case ncclBfloat16:
#endif
		// case ncclFloat16:
		return 2;
	case ncclInt:
	case ncclFloat:
#if NCCL_MAJOR >= 2
	// case ncclInt32:
	case ncclUint32:
		// case ncclFloat32:
#endif
		return 4;
	case ncclInt64:
	case ncclUint64:
	case ncclDouble:
		// case ncclFloat64:
		return 8;
	default:
		return 0;
	}
}

extern int test_ncclVersion; // init'd with ncclGetVersion()
const int test_opNumMax = (int)ncclNumOps + (NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0) ? 1 : 0);
extern int test_opnum;
extern int test_typenum;
extern ncclDataType_t test_types[ncclNumTypes];
extern const char* test_typenames[ncclNumTypes];
extern ncclRedOp_t test_ops[];
extern const char* test_opnames[];

#endif
