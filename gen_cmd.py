# srun --partition=caif_rd -n16 --ntasks-per-node=8 -w SH-IDC1-10-140-1-175,SH-IDC1-10-140-1-176     --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n16 -N2 --ntasks-per-node=8 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n16 -N2 --ntasks-per-node=8 -w SH-IDC1-10-140-1-175,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1

# srun --partition=caif_rd -n2 -N2 --ntasks-per-node=1 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n8 -N2 --ntasks-per-node=4 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# ps -efww|grep -w 'rdma-tutorial'|grep -v grep|awk '{print $2}'|xargs kill -9

# srun --partition=caif_rd  -w SH-IDC1-10-140-0-178 --preempt ./0_ib_launch.conf
# srun --partition=caif_rd  -w SH-IDC1-10-140-0-207 --preempt ./1_ib_launch.conf
# export LD_LIBRARY_PATH=/mnt/cache/wangguoteng.p/nccl/build_master/lib:/mnt/cache/wangguoteng.p/SimpleRDMA/lib:/mnt/cache/share/cuda-11.3/lib64 
import os
import stat
# 2052096
"""
export NCCL_NTHREADS=512
# export NCCL_MAX_NCHANNELS=512
# export NCCL_MIN_NCHANNELS=32
export NCCL_BUFFSIZE=41943040
export NCCL_IB_HCA=mlx5_0
export NCCL_PROTO=Simple
export NCCL_P2P_NET_CHUNKSIZE=262144
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1

export NCCL_IB_HCA=mlx5_0,mlx5_bond_0

unset NCCL_NTHREADS
unset NCCL_MAX_NCHANNELS
unset NCCL_MIN_NCHANNELS
unset NCCL_BUFFSIZE
unset NCCL_IB_HCA
"""

node1 = "SZ-OFFICE2-172-20-21-185"
node2 = "SZ-OFFICE2-172-20-21-189"
USE_BASH = True
path = "/mnt/cache/wangguoteng.p/SimpleRDMA/"
exec_bin = path + "rdma-tutorial"

# SH-IDC1-10-140-0-31
# node_list = ["SZ-OFFICE2-172-20-21-185", "SZ-OFFICE2-172-20-21-189"]
node_list = ["SH-IDC1-10-140-0-149", "SH-IDC1-10-140-0-150"]
# node_list = ["SH-IDC1-10-140-0-31", "SH-IDC1-10-140-0-31"]
# int((15 * 1024 * 1024) / (15*8))

USE_NSYS = False
NSYS_REPORT_NAME="ngpus-new-pcie-report"
if USE_NSYS:
    nsys = "/mnt/petrelfs/caifcicd/dev/nsys/opt/nvidia/nsight-systems/2022.3.4/bin/nsys profile --stats=true --force-overwrite=true  --trace=cuda  --sample=cpu -o {}".format(NSYS_REPORT_NAME)
else:
    nsys = ""

p2p=True
if p2p:
    nRanks = 2
    taskPerNode = [1, 1]
    msg_size = int(4)
    # print("msg_size: {} MB".format(msg_size/1024.0/1024.0))
    print("msg_size: {} B".format(msg_size))
else:
    nRanks = 16
    taskPerNode = [8, 8]
    #  nccl p2p : X/8 = msg_size 
    #  nccl p2p : X/8 = msg_size 
    #  nccl all2all : x = msg_size * 8  * 8 
    #  nccl all2all : x / 64 = msg_size 
    msg_size = int((1 * 1024 * 1024 / 64))
    # msg_size = int((1 * 1024 * 1024) / (8*8))
    
    # ibverbs all2all
    # x = msg_size * 15 * 8
    msg_size = int(16 * 1024 * 1024 / 120)
    print("msg_size: {} MB".format(msg_size/1024.0/1024.0))
    

nodeNum = len(node_list)
rank = 0



if __name__ == "__main__":
    if not USE_BASH:
        with open("ib_launch.conf", "w") as fb:

            for rank in range(nRanks):
                sstr = str(rank) + ' ' + exec_bin + ' ' + str(nRanks) + \
                    ' ' + str(rank) + ' ' + str(msg_size) + ' '
                for i in range(nRanks // 2):
                    sstr += (node1 + ' ')
                for i in range(nRanks // 2):
                    sstr += (node2 + ' ')
                fb.writelines(sstr + '\n')
    else:

        files = ["{}_ib_launch.conf".format(i) for i in range(nodeNum)]
        for nodeCount, f in enumerate(files):
            with open(f, "w") as fb:
                fb.writelines("#!/bin/bash\n")
                fb.writelines("hostname\n")

                for i in range(taskPerNode[nodeCount]):
                    if rank == 0:
                        sstr = nsys + ' ' + exec_bin + ' ' + \
                            str(nRanks) + ' ' + str(rank) + \
                            ' ' + str(msg_size) + ' ' + str(taskPerNode[nodeCount]) + ' '
                    else:
                        sstr = exec_bin + ' ' + \
                            str(nRanks) + ' ' + str(rank) + \
                            ' ' + str(msg_size) + ' ' + str(taskPerNode[nodeCount]) + ' '
                    for j in range(nodeNum):
                        for p in range(taskPerNode[j]):
                            sstr += (node_list[j] + ' ')

                    if i != taskPerNode[nodeCount] - 1:
                        fb.writelines(sstr + ' &' + '\n')
                    else:
                        fb.writelines(sstr + '\n')

                    rank += 1
                fb.writelines(
                    "sleep 20 && ps -efww|grep -w 'rdma-tutorial'|grep -v grep|awk '{print $2}'|xargs kill -9\n"
                )
            os.chmod(f, stat.S_IRWXU)
