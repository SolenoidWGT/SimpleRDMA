# srun --partition=caif_rd -n16 --ntasks-per-node=8 -w SH-IDC1-10-140-1-175,SH-IDC1-10-140-1-176     --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n16 -N2 --ntasks-per-node=8 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n16 -N2 --ntasks-per-node=8 -w SH-IDC1-10-140-1-175,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1

# srun --partition=caif_rd -n2 -N2 --ntasks-per-node=1 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
# srun --partition=caif_rd -n8 -N2 --ntasks-per-node=4 -w SH-IDC1-10-140-0-166,SH-IDC1-10-140-1-176 --preempt  -l --multi-prog ib_launch.conf >log.log 2>&1
#
msg_size = 16
nRanks = 16
node1 = "SH-IDC1-10-140-1-175"
node2 = "SH-IDC1-10-140-1-176"
bin = "/mnt/cache/wangguoteng.p/RDMA-Tutorial/rdma-tutorial"

with open("ib_launch.conf", "w") as fb:

    for rank in range(nRanks):
        sstr = str(rank) + ' ' + bin + ' ' + str(nRanks) + ' ' + str(
            rank) + ' ' + str(msg_size) + ' '
        for i in range(nRanks / 2):
            sstr += (node1 + ' ')
        for i in range(nRanks / 2):
            sstr += (node2 + ' ')
        fb.writelines(sstr + '\n')
