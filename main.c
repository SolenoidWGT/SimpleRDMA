#define _GNU_SOURCE         /* See feature_test_macros(7) */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

#include "debug.h"
#include "config.h"
#include "ib.h"
#include "setup_ib.h"
#include "client.h"
#include "server.h"

FILE *log_fp = NULL;

char *SERVER_PORT = "12345";
char *CLIENT_PORT = "12346";

int init_env();
void destroy_env();

struct ibv_mr * (*ibv_internal_reg_mr_iova2)(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, int access);

int main(int argc, char *argv[])
{
    int ret = 0;

    // if (argc != 5)
    // {
    //     printf("Usage: %s config_file sock_port\n", argv[0]);
    //     return 0;
    // }

    // ret = parse_config_file(argv[1]);
    // check(ret == 0, "Failed to parse config file");

    config_info.is_server = (bool)atoi(argv[1]);
    config_info.msg_size = 16;
    config_info.num_concurr_msgs = 8;
    config_info.num_servers = 1;
    config_info.num_clients = 1;

    config_info.servers = (char **)malloc(sizeof(char *));
    config_info.servers[0] = (char *)malloc(128);
    config_info.clients = (char **)malloc(sizeof(char *));
    config_info.clients[0] = (char *)malloc(128);
    sprintf(config_info.servers[0], "%s", argv[3]);
    sprintf(config_info.clients[0], "%s", argv[4]);

    if (config_info.is_server)
        config_info.sock_port = SERVER_PORT;
    else
        config_info.sock_port = CLIENT_PORT;

    config_info.rank = 0;
    ret = init_env();
    check(ret == 0, "Failed to init env");

    ret = setup_ib();
    check(ret == 0, "Failed to setup IB");

    if (config_info.is_server)
    {
        ret = run_server();
    }
    else
    {
        ret = run_client();
    }
    check(ret == 0, "Failed to run workload");

error:
    // if (ibvhandle != NULL) dlclose(ibvhandle);
    close_ib_connection();
    destroy_env();
    return ret;
}

int init_env()
{
    char fname[64] = {'\0'};

    if (config_info.is_server)
    {
        sprintf(fname, "server[%d].log", config_info.rank);
    }
    else
    {
        sprintf(fname, "client[%d].log", config_info.rank);
    }
    log_fp = fopen(fname, "w");
    check(log_fp != NULL, "Failed to open log file");

    log(LOG_HEADER, "IB Echo Server");
    print_config_info();

    return 0;
error:
    return -1;
}

void destroy_env()
{
    log(LOG_HEADER, "Run Finished");
    if (log_fp != NULL)
    {
        fclose(log_fp);
    }
}
