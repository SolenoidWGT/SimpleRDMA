# srun --partition=caif_rd -n16 --ntasks-per-node=8 -w SH-IDC1-10-140-1-175,SH-IDC1-10-140-1-176     --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n16 -N2 --ntasks-per-node=8 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n16 -N2 --ntasks-per-node=8 -w SH-IDC1-10-140-1-175,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1

# srun --partition=caif_rd -n2 -N2 --ntasks-per-node=1 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n8 -N2 --ntasks-per-node=4 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# ps -efww|grep -w 'rdma-tutorial'|grep -v grep|awk '{print $2}'|xargs kill -9

import os
import stat
# 2052096

node1 = "SZ-OFFICE2-172-20-21-185"
node2 = "SZ-OFFICE2-172-20-21-189"
USE_BASH = True
exec_bin = "/mnt/lustre/wangguoteng2/SimpleRDMA/rdma-tutorial"


node_list = ["SZ-OFFICE2-172-20-21-185", "SZ-OFFICE2-172-20-21-189"]
msg_size = 2052096
nRanks = 2
nodeNum = len(node_list)
taskPerNode = nRanks // nodeNum
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
        for f in files:
            with open(f, "w") as fb:
                fb.writelines("#!/bin/bash\n")

                for i in range(taskPerNode):
                    sstr = exec_bin + ' ' + \
                        str(nRanks) + ' ' + str(rank) + \
                        ' ' + str(msg_size) + ' ' + str(taskPerNode) + ' '
                    for j in range(nodeNum):
                        for p in range(taskPerNode):
                            sstr += (node_list[j] + ' ')

                    if i != taskPerNode - 1:
                        fb.writelines(sstr + ' &' + '\n')
                    else:
                        fb.writelines(sstr + '\n')

                    rank += 1
                fb.writelines(
                    "sleep 20 && ps -efww|grep -w 'rdma-tutorial'|grep -v grep|awk '{print $2}'|xargs kill -9\n")
            os.chmod(f, stat.S_IXOTH)
