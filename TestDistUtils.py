import math
import torch
import sys, os, time
import argparse



from minigpt4.common.dist_utils import *
from deepspeed.accelerator import get_accelerator
from deepspeed.comm import TorchBackend

DEFAULT_WARMUPS = 5
DEFAULT_TRIALS = 50
DEFAULT_TYPE = 'float'
DEFAULT_BACKEND = get_accelerator().communication_backend_name()
DEFAULT_UNIT = 'Gbps'
DEFAULT_DIST = 'deepspeed'
DEFAULT_MAXSIZE = 24
TORCH_DISTRIBUTED_DEFAULT_PORT = 29500

def get_bw(comm_op, size, duration, args):
    n = dist.get_world_size()
    tput = 0
    busbw = 0
    if comm_op == "all_to_all":
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_gather":
        size *= n
        tput = (size / duration)
        busbw = (size / duration) * ((n - 1) / n)
    elif comm_op == "all_reduce":
        tput = (size * 2 / duration)
        busbw = (size / duration) * (2 * (n - 1) / n)
    elif comm_op == "pt2pt" or comm_op == "broadcast":
        tput = (size / duration)
        busbw = tput
    else:
        print_rank_0("wrong comm_op specified")
        exit(0)

    if args.bw_unit == 'Gbps':
        tput *= 8
        busbw *= 8

    return tput, busbw

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def get_metric_strings(args, tput, busbw, duration):
    duration_ms = duration * 1e3
    duration_us = duration * 1e6
    tput = f'{tput / 1e9:.3f}'
    busbw = f'{busbw /1e9:.3f}'

    if duration_us < 1e3 or args.raw:
        duration = f'{duration_us:.3f}'
        if not args.raw:
            duration += ' us'
    else:
        duration = f'{duration_ms:.3f} ms'
    return tput, busbw, duration

def max_numel(comm_op, dtype, mem_factor, local_rank, args):
    dtype_size = _element_size(dtype)
    max_memory_per_gpu = get_accelerator().total_memory(local_rank) * mem_factor
    if comm_op == 'all_reduce' or comm_op == 'pt2pt' or comm_op == 'broadcast':
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
    elif comm_op == 'all_gather':
        # all_gather performance is lower for non-powers of two, and the output buffer size scales with world size
        # Therefore, divide by world size and round down to nearest power of 2
        elements_per_gpu = int(max_memory_per_gpu // dtype_size // dist.get_world_size())
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    elif comm_op == 'all_to_all':
        # Number of elements must be divisible by world_size
        # all_to_all performance is lower for non-powers of two. Round down like all_gather.
        elements_per_gpu = int(max_memory_per_gpu // dtype_size)
        elements_per_gpu = int(dist.get_world_size() * round(elements_per_gpu / dist.get_world_size()))
        elements_per_gpu = int(pow(2, int(math.log(elements_per_gpu, 2))))
    else:
        print(f"This communication operation: {comm_op} is not supported yet")
        exit(0)
    return elements_per_gpu

def timed_all_gather(input, output, args):
    if args.dist == 'torch':
        import torch.distributed as dist

        all_gather_func = TorchBackend.get_all_gather_function()
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

        all_gather_func = dist.allgather_fn

    sync_all()
    # Warmups, establish connections, etc.
    for i in range(args.warmups):
        all_gather_func(output, input, group=None, async_op=args.async_op)
    sync_all()

    # time the actual comm op trials times and average it
    pre = time.perf_counter()
    for i in range(args.trials):
        all_gather_func(output, input, group=None, async_op=args.async_op)
    sync_all()
    duration = time.perf_counter() - pre

    # maintain and clean performance data
    avg_duration = duration / args.trials
    size = input.element_size() * input.nelement()
    tput, busbw = get_bw('all_gather', size, avg_duration, args)
    tput_str, busbw_str, duration_str = get_metric_strings(args, tput, busbw, avg_duration)
    desc = f'{input.nelement()}x{input.element_size()}'

    if not args.raw:
        size = convert_size(size)

    print_rank_0(f"{size:<20} {desc:25s} {duration_str:20s} {tput_str:20s} {busbw_str:20s}")



def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f'expected torch.dtype, but got {type(dtype)}')

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3

def run_all_gather(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # Prepare benchmark header
   # print_header(args, 'all_gather')
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    if args.scan:
        # Create list of message sizes
        M_LIST = []
        for x in (2**p for p in range(1, args.maxsize)):
            M_LIST.append(x)

        sync_all()
        # loop over various tensor sizes
        for M in M_LIST:
            global_rank = dist.get_rank()
            try:
                mat = torch.ones(world_size, M,
                                 dtype=getattr(torch, args.dtype)).to(get_accelerator().device_name(local_rank))
                sync_all()
                input = ((mat.mul_(float(global_rank))).view(-1))
                # Delete original mat to avoid OOM
                del mat
                get_accelerator().empty_cache()
                output = torch.zeros(input.nelement() * world_size,
                                     dtype=getattr(torch, args.dtype)).to(get_accelerator().device_name(local_rank))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    if dist.get_rank() == 0:
                        print('WARNING: Ran out of GPU memory. Exiting comm op.')
                    sync_all()
                    break
                else:
                    raise e
            sync_all()
            timed_all_gather(input, output, args)
    else:
        # all_gather_into_tensor saves memory
        if ((args.dist == 'torch' or args.dist == 'deepspeed') and dist.has_all_gather_into_tensor()):
            mem_factor = args.mem_factor + 0.2
        else:
            mem_factor = args.mem_factor
        # Send the biggest message size our GPUs can fit. If you're facing OOM errors, reduce the mem_factor
        sync_all()
        elements_per_gpu = max_numel(comm_op='all_gather',
                                     dtype=getattr(torch, args.dtype),
                                     mem_factor=mem_factor,
                                     local_rank=local_rank,
                                     args=args)
        try:
            mat = torch.ones(elements_per_gpu, dtype=getattr(torch,
                                                             args.dtype)).to(get_accelerator().device_name(local_rank))
            # multiply each GPU's tensor by the rank to ease debugging
            input = ((mat.mul_(float(global_rank))).view(-1))
            # Delete original mat to avoid OOM
            del mat
            get_accelerator().empty_cache()
            output = torch.zeros(elements_per_gpu * world_size,
                                 dtype=getattr(torch, args.dtype)).to(get_accelerator().device_name(local_rank))
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if dist.get_rank() == 0:
                    print('WARNING: Ran out of GPU memory. Try to reduce the --mem-factor argument!')
                sync_all()
                return
            else:
                raise e

        sync_all()
        timed_all_gather(input, output, args)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help='Number of timed iterations')
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS, help='Number of warmup (non-timed) iterations')
    parser.add_argument("--maxsize", type=int, default=24, help='Max message size as a power of 2')
    parser.add_argument("--async-op", action="store_true", help='Enables non-blocking communication')
    parser.add_argument("--bw-unit", type=str, default=DEFAULT_UNIT, choices=['Gbps', 'GBps'])
    parser.add_argument("--backend",
                        type=str,
                        default=DEFAULT_BACKEND,
                        choices=['nccl', 'ccl', 'mpi'],
                        help='Communication library to use')
    parser.add_argument("--dist",
                        type=str,
                        default=DEFAULT_DIST,
                        choices=['deepspeed', 'torch'],
                        help='Distributed DL framework to use')
    parser.add_argument("--scan", action="store_true", help='Enables scanning all message sizes')
    parser.add_argument("--raw", action="store_true", help='Print the message size and latency without units')
    parser.add_argument("--all-gather", action="store_true", help='Run all_gather')
    parser.add_argument("--dtype", type=str, default=DEFAULT_TYPE, help='PyTorch tensor dtype')
    parser.add_argument("--mem-factor",
                        type=float,
                        default=.3,
                        help='Proportion of max available GPU memory to use for single-size evals')
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    rank = args.local_rank
    args.dist_backend = args.dist
    init_processes(args=args)
    run_all_gather(local_rank=rank, args=args)
