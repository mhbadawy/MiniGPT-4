"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import deepspeed
from deepspeed.accelerator import get_accelerator

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_processes
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    print(cfg)
    # init_processes(cfg.run_cfg)
    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    # task = tasks.setup_task(cfg)
    # datasets = task.build_datasets(cfg)
    # model = task.build_model(cfg)

    # num_parameters = 0
    # p_wd, p_non_wd = [], []
    # for n, p in model.named_parameters():
    #     if not p.requires_grad:
    #         continue  # frozen weights
    #     print(n)
    #     if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
    #         p_non_wd.append(p)
    #     else:
    #         p_wd.append(p)
    #     num_parameters += p.data.nelement()
    # optim_params = [
    #     {
    #         "params": p_wd,
    #         "weight_decay": float(cfg.run_cfg.weight_decay),
    #     },
    #     {"params": p_non_wd, "weight_decay": 0},
    # ]
    # model_engine, optimizer, __ , __ = deepspeed.initialize(
    #     args=args, model=model, model_parameters=optim_params, config=ds_config)
    # print(model_engine)
    # print(optimizer)
    #
    #
    # if cfg.run_cfg.wandb_log:
    #     wandb.login()
    #     wandb.init(project="minigptv", name=cfg.run_cfg.job_name)
    #     wandb.watch(model)
    #
    # runner = get_runner_class(cfg)(
    #     cfg=cfg, job_id=job_id, task=task, model=model_engine, datasets=datasets, optimizer=optimizer
    # )
   # runner.train()


if __name__ == "__main__":
    main()
