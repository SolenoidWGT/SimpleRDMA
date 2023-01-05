import math

nRanks = 16
p2p = []
r = 2

abc = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'b', 'C', 'D', 'E', 'F']
z_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

for i in range(nRanks):
    p2p.append([])
    for j in range(nRanks):
        if (i <= 7 and j <= 7) or (i > 7 and j > 7):
            p2p[i].append(1)
        else:
            p2p[i].append(0)

print(p2p)


def f(n, r_x, w):
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'b', 'C', 'D', 'E', 'F']
    b = []
    while True:
        s = n // r_x
        y = n % r_x
        b = b + [y]
        if s == 0:
            break
        n = s
    b.reverse()
    if len(b) < w:
        b = [0] * (w - len(b)) + b

    return [a[b] for b in b]


allRankIntra = 0
allRankInter = 0
intra_subphase_list = [0 for i in range(32)]
inter_subphase_list = [0 for i in range(32)]
w = int(math.log(nRanks, r))
print("w:{}".format(w))
for rank in range(nRanks):
    x = 0
    r_x = 1
    perRankIntra = 0
    perRankInter = 0
    roundCount = 0
    while r_x < nRanks:
        displs = []
        z_list = []

        for z in range(1, r):
            z_name = abc[z]
            blockSendRecv_list = []
            sendRecv_set = set()
            for block in range(1, nRanks, 1):
                bpof2 = f(block, r, w)
                # print("bpof2:{}, r_x:{}".format(bpof2, r_x))
                # if block & r_x:
                if bpof2[x] == z_name:
                    nextRingPeer = (rank + z_10[z] * r_x) % nRanks
                    preRingPeer = (rank - z_10[z] * r_x + nRanks) % nRanks
                    blockSendRecv_list.append(
                        (block, nextRingPeer, preRingPeer))

            for block_3 in blockSendRecv_list:
                if block_3[1] not in sendRecv_set:
                    perRankIntra += p2p[rank][nextRingPeer]
                    perRankInter += (1 - p2p[rank][nextRingPeer])
                    intra_subphase_list[r_x] += p2p[rank][nextRingPeer]
                    inter_subphase_list[r_x] += (1 - p2p[rank][nextRingPeer])
                    sendRecv_set.add(block_3[1])
                # print(
                #     "{} rank round:{}, block:{}, intra-node:{}, inter-node:{}"
                #     .format(rank, r_x, block, intra, 2 - intra))

            print(
                "[Rank-[{}], subphase-[{}], steps-[{}], blockNeedSend:[{}], displs:{}"
                .format(rank, r_x, z, len(blockSendRecv_list),
                        blockSendRecv_list))

        r_x = r_x * r
        x += 1
        roundCount += 1
    print("{} rank: intra-node:{}, inter-node:{}".format(
        rank, perRankIntra, perRankInter))
    allRankIntra += perRankIntra
    allRankInter += perRankInter
    print("+++++++++++++++++++++++++++++++++++++++++++++")

print("ALL intra:{}, inter:{}".format(allRankIntra, allRankInter))
for i in range(roundCount):
    print("Round:{}, intra:{}, inter:{}".format(r**i,
                                                intra_subphase_list[r**i],
                                                inter_subphase_list[r**i]))
