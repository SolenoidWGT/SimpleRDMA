import torch
import torch.distributed as dist
import multiprocessing as mp
import time
import argparse
import os

init_method="tcp://10.140.0.180:12347"
world_size = 16

def shape2ByteSize(tensor):
    elem_size = tensor.element_size()
    elem_num = tensor.numel()
    return elem_size * elem_num

def init_all_to_all_single(tensor_size=5344 * 1536):
    rank = dist.get_rank()
    gpu = f"cuda:{torch.cuda.current_device()}"
    tensor = torch.arange(tensor_size, device=gpu, dtype=torch.float32) + rank * tensor_size
    tensor = tensor.view(dist.get_world_size(), -1)
    output = torch.empty_like(tensor)
    return tensor, output

def nccl_init(rank, base):
    print("rank: {}, base:{}".format(rank, base))
    # os.environ['NCCL_DEBUG'] = "INFO"
    # os.environ['NCCL_IB_HCA'] = "mlx5_0"
    # os.environ['NCCL_P2P_LEVEL'] = "SYS"
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method)
    local_rank = rank - base
    torch.cuda.set_device(f"cuda:{local_rank}")
    tensor, output = init_all_to_all_single()
    print("tensor shape[{}], output shape[{}], elem size[{}], dtype[{}]".format(tensor.shape, output.shape, tensor.element_size(), tensor.dtype))
    print("tensor byte size [{}] MB, output byte size [{}] MB".format(shape2ByteSize(tensor) / (1024*1024), shape2ByteSize(output)  / (1024*1024)))

    # warm up
    start = time.time()
    dist.all_to_all_single(output, tensor)
    torch.cuda.synchronize(local_rank)
    # dist.all_reduce(tensor)
    torch.cuda.synchronize(local_rank)
    end = time.time()
    print("all2all-[{}] the first iter use {:.2f} ms".format(world_size, (end-start)*1000))

    time_list  = []
    for i in range(20):
        start = time.time()
        dist.all_to_all_single(output, tensor)
        torch.cuda.synchronize(local_rank)
        # dist.all_reduce(tensor)
        end = time.time()
        time_list.append((end-start)*1000)
    
    avg_time = sum(time_list) / 20
    print("all2all-[{}] skip first iter 20 use {:.2f} ms".format(world_size, avg_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test torch rpc')
    parser.add_argument('--base', type=int)
    args, _ = parser.parse_known_args()
    base = int(args.base)

    ctx = mp.get_context("spawn")
    local_ranks = [(i+base, base) for i in range(8)]

    with ctx.Pool(processes=8) as pool:
        pool.starmap(nccl_init, local_ranks)

