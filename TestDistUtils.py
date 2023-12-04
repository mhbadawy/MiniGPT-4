# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import sys, os, time



from minigpt4.common.dist_utils import *
from deepspeed.accelerator import get_accelerator
from deepspeed.comm import TorchBackend


def run_all_gather(local_rank, args):
    if args.dist == 'torch':
        import torch.distributed as dist
    elif args.dist == 'deepspeed':
        import deepspeed.comm as dist

    # Prepare benchmark header
    print_header(args, 'all_gather')
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
    init_processes(local_rank=rank, args=args)
    run_all_gather(local_rank=rank, args=args)
