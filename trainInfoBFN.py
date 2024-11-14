# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import logging
import math
from collections import defaultdict
import neptune.new as neptune
from pathlib import Path
from typing import Optional, Tuple
import os

import torch

from accelerate import Accelerator,InitProcessGroupKwargs
from datetime import timedelta
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from rich.logging import RichHandler
from rich.progress import Progress
from torch import nn, optim
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

from model import BFN
from utils_train import (
    seed_everything, log_cfg,
    checkpoint_training_state,
    init_checkpointing,
    log,
    update_ema,
    ddict,
    make_infinite,
    make_progress_bar, make_config, make_dataloaders, make_bfn,
)

from accelerate import DistributedDataParallelKwargs
import time


# def setup(cfg) -> Tuple[nn.Module, dict, optim.Optimizer]:
#     """Create the model, dataloader and optimizer"""
#     dataloaders = make_dataloaders(cfg)
#     # print("-------------Dataloader-------------")
#     # for x  in dataloaders["train"]:
#     #     print(x.shape)
#     #     #print(y.shape)
#     #     break
#     # assert False
#     # print("-------------Dataloader-------------")
#     model = make_bfn(cfg.model)
#     if "weight_decay" in cfg.optimizer.keys() and hasattr(model.net, "get_optim_groups"):
#         params = model.get_optim_groups(cfg.optimizer.weight_decay)
#     else:
#         params = model.parameters()
#     # Instantiate the optimizer using the hyper-parameters in the config
#     optimizer = optim.AdamW(params=params, **cfg.optimizer)
#     return model, dataloaders, optimizer
import json
def load_training_info(info_path):
    """Load training information, such as iteration count and run ID, from a JSON file if it exists."""
    if os.path.exists(info_path):
        with open(info_path, "r") as file:
            info = json.load(file)
        return info['step'], info['run_id']
    else:
        return None


# def setup(cfg) -> Tuple[nn.Module, dict, optim.Optimizer]:
#     """Create the model, dataloader and optimizer"""
#     dataloaders = make_dataloaders(cfg)
#     # print("-------------Dataloader-------------")
#     # for x  in dataloaders["train"]:
#     #     print(x.shape)
#     #     #print(y.shape)
#     #     break
#     # assert False
#     # print("-------------Dataloader-------------")
#
#     model = make_bfn(cfg.model)
#     checkpoint_path = os.path.join(cfg.training.checkpoint_dir, cfg.training.name, "last", "last.pt")
#     #checkpoint_path = "./checkpoints/0916-0854_ffhq100e/last/last.pt"
#     if checkpoint_path:
#         # Load the checkpoint
#         model.load_state_dict(torch.load(checkpoint_path))
#     if "weight_decay" in cfg.optimizer.keys() and hasattr(model.net, "get_optim_groups"):
#         params = model.get_optim_groups(cfg.optimizer.weight_decay)
#     else:
#         params = model.parameters()
#     # Instantiate the optimizer using the hyper-parameters in the config
#     optimizer = optim.AdamW(params=params, **cfg.optimizer)
#     return model, dataloaders, optimizer
def setup(cfg) -> Tuple[nn.Module, dict, optim.Optimizer]:
    """Create the model, dataloader, and optimizer"""
    dataloaders = make_dataloaders(cfg)

    model = make_bfn(cfg.model)

    # Ensure the checkpoint directory exists
    checkpoint_dir = os.path.join(cfg.training.checkpoint_dir, cfg.training.name, "last")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "last.pt")

    if os.path.exists(checkpoint_path):
        # Load the checkpoint if it exists
        model.load_state_dict(torch.load(checkpoint_path))

    if "weight_decay" in cfg.optimizer.keys() and hasattr(model.net, "get_optim_groups"):
        params = model.get_optim_groups(cfg.optimizer.weight_decay)
    else:
        params = model.parameters()

    # Instantiate the optimizer using the hyper-parameters in the config
    optimizer = optim.AdamW(params=params, **cfg.optimizer)

    return model, dataloaders, optimizer
@torch.no_grad()
def validate(
        cfg,
        model: BFN,
        ema_model: nn.Module,
        val_dataloader: DataLoader,
        step: int,
        run: "neptune.Run",

        pbar: Optional[Progress],
        best_val_loss: float,
        checkpoint_root_dir: Optional[Path],
        accelerator: Accelerator,
) -> float:
    """Evaluate model acceleraon validation data and save checkpoint if loss improves"""
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[accelerator.mixed_precision]
    model_to_eval = ema_model if ema_model is not None else model
    model_to_eval.eval()
    pbar = pbar or Progress()
    max_steps = cfg.max_val_batches if cfg.max_val_batches > 0 else len(val_dataloader)
    val_id = pbar.add_task("Validating", visible=True, total=cfg.val_repeats * max_steps, transient=True, loss=math.nan)


    traningloss,loss, count, = 0.0,0.0, 0
    for i in range(cfg.val_repeats):
        for idx, eval_batch in enumerate(val_dataloader):
            if cfg.dataset in ['bin_mnist', 'celeba']:
                eval_batch = eval_batch[0]
                #print(eval_batch.shape)
                #assert False
            enabled = True if dtype in [torch.float16, torch.bfloat16] else False
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=dtype, enabled=enabled):
                Renloss, KL_loss, MMD_loss,whole_klloss = model_to_eval(eval_batch.to(accelerator.device))
                traningloss = traningloss + Renloss + whole_klloss
                count += 1
            pbar.update(val_id, advance=1, loss=traningloss / count)
            if (idx + 1) >= max_steps:
                break
    loss = traningloss /count
    pbar.remove_task(val_id)
    log(run["metrics"]["val"]["loss"], loss, step)

    if checkpoint_root_dir is not None and (loss < best_val_loss or math.isinf(best_val_loss)):
        #logger.info(f"loss improved: new value is {loss}")
        step_checkpoint_path = checkpoint_root_dir / "best"
        run_id = "B128_e5" if isinstance(run, defaultdict) else run["sys"]["id"].fetch()
        checkpoint_training_state(step_checkpoint_path, accelerator, ema_model, step, run_id)
        run["metrics/best/loss/metric"] = loss
        run["metrics/best/loss/step"] = step

    model.train()
    return loss


def train(
        cfg,
        accelerator: Accelerator,
        model: BFN,
        ema_model: Optional[nn.Module],
        dataloaders: dict,
        optimizer: optim.Optimizer,
        run: "neptune.Run"
):
    is_main = accelerator.is_main_process
    pbar = make_progress_bar(is_main)
    #run_id = "debug" if isinstance(run, defaultdict) else run["sys"]["id"].fetch()
    run_id = cfg.run_id  # 使用加载的运行ID
    ##train_id = pbar.add_task(f"Training {run_id}", start=cfg.start_step, total=cfg.n_training_steps, loss=math.nan)
    train_id = pbar.add_task(f"Training {run_id}", completed=cfg.start_step, total=cfg.n_training_steps, loss=math.nan)


    #train_id = pbar.add_task(f"Training {run_id}", start=cfg.start_step, total=cfg.n_training_steps, loss=math.nan)
    #exp_name = time.strftime("%m%d-%H%M") + "_" + cfg.name
    exp_name = cfg.name
    checkpoint_root_dir = init_checkpointing(cfg.checkpoint_dir, exp_name) if is_main else None
    best_val_loss = math.inf

    train_iter = make_infinite(dataloaders["train"])
    model.train()
    #accum_steps = 2
    with pbar:
        for step in range(cfg.start_step, cfg.n_training_steps + 1):
            step_loss,step_KLloss,step_MMD,Step_WhoKL,Step_WholREc = 0.0,0.0,0.0,0.0,0.0
            for _ in range(cfg.accumulate):
                with accelerator.accumulate(model):
                    if cfg.dataset in ['celeba', "bin_mnist", "cifar10", "bfashion_mnist"]:
                        train_batch, target = next(train_iter)
                        # print(train_batch.shape)
                        # assert False
                        # train_batch = train_batch[0]
                    elif cfg.dataset in ['dsprites']:
                        train_batch, latents, target = next(train_iter)

                    elif cfg.dataset in ['shapes3d']:
                        train_batch, _, target, _ = next(train_iter)
                    elif cfg.dataset in ['ffhq']:
                        train_batch = next(train_iter)

                    loss, KL_loss, MMD_loss,whole_klloss,Reconloss = model(train_batch)

                    accelerator.backward(loss+whole_klloss)

                    if accelerator.sync_gradients and cfg.grad_clip_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    #if (step+1) % accum_steps == 0 or step == cfg.n_training_steps+1:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                step_loss += loss.item()
                step_KLloss += KL_loss.item()
                step_MMD += MMD_loss.item()
                Step_WhoKL += whole_klloss.item()
                Step_WholREc += Reconloss.item()

            update_ema(ema_model, model, cfg.ema_decay)

            if is_main and (step % cfg.checkpoint_interval == 0):
                checkpoint_training_state(checkpoint_root_dir /"last", accelerator, ema_model, step, run_id)
                run["checkpoints/last"].track_files(str(checkpoint_root_dir / "last"))
                #run.upload_files(str(checkpoint_root_dir / "last"))

            log(run["metrics"]["train"]["loss"], step_loss / cfg.accumulate, step, is_main and step % cfg.log_interval == 0)
            log(run["metrics"]["train"]["KL_loss"], step_KLloss / cfg.accumulate, step, is_main and step % cfg.log_interval == 0)
            log(run["metrics"]["train"]["MMD_loss"], step_MMD / cfg.accumulate, step, is_main and step % cfg.log_interval == 0)
            log(run["metrics"]["train"]["whole_klloss"], Step_WhoKL / cfg.accumulate, step, is_main and step % cfg.log_interval == 0)
            log(run["metrics"]["train"]["recloss"], Step_WholREc / cfg.accumulate, step,
                is_main and step % cfg.log_interval == 0)
            #log(run["metrics"]["epoch"], step // len(dataloaders["train"]), step, is_main)
            # print("loss",step_loss / cfg.accumulate)
            # print("KL_loss",step_KLloss / cfg.accumulate)
            # print("MMD_loss",step_MMD / cfg.accumulate)
            # print("whole_klloss",Step_WhoKL / cfg.accumulate)
            #assert False

            if is_main and (step % cfg.val_interval == 0) and "val" in dataloaders:
                val_loss = validate(
                    cfg=cfg,
                    model=model,
                    ema_model=ema_model,
                    val_dataloader=dataloaders["val"],
                    step=step,
                    run=run,
                    pbar=pbar,
                    best_val_loss=best_val_loss,
                    checkpoint_root_dir=checkpoint_root_dir,
                    accelerator=accelerator,
                )
                best_val_loss = min(val_loss, best_val_loss)

            pbar.update(train_id, advance=1, loss=loss.item())


def main(cfg):
    # Configure the distributed process group
    # timeout_seconds = 90 * 60  # Set to 1 hour, for example
    # kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=timeout_seconds))



    # Check if the training information was successfully loaded



    # Initialize the Accelerator with custom process group initialization parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # #accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    acc = Accelerator(kwargs_handlers=[ddp_kwargs],gradient_accumulation_steps=cfg.training.accumulate)

    seed_everything(cfg.training.seed)
    # log_dir = 'logs'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # log_file = os.path.join(log_dir, cfg.training.name+'.log')

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
        #filename=log_file,
    )
    logger = get_logger(__name__)
    logger.info(f"Seeded everything with seed {cfg.training.seed}", main_process_only=True)

    with acc.main_process_first():
        model, dataloaders, optimizer = setup(cfg)
        #acc.wait_for_everyone()
    ema = copy.deepcopy(model) if acc.is_main_process and cfg.training.ema_decay > 0 else None  # EMA on main proc only
    model, optimizer, dataloaders["train"] = acc.prepare(model, optimizer, dataloaders["train"])
    #acc.wait_for_everyone()
    run = ddict()
    if acc.is_main_process:
        ema.to(acc.device)
        try:
            if cfg.meta.neptune:
                import neptune
                run = neptune.init_run(project=cfg.meta.neptune, name=cfg.training.name,mode="debug" if cfg.meta.debug else None, api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMmU1NDZiNi1jYTM4LTQ2ZWUtYjczMi1jMDNmYTkwODQ2MzEifQ==")
                run["accelerate"] = dict(amp=acc.mixed_precision, nproc=acc.num_processes)
                log_cfg(cfg, run)
        except ImportError:
            logger.info("Did not find neptune installed. Logging will be disabled.")

    training_info = load_training_info(
        os.path.join(cfg.training.checkpoint_dir, cfg.training.name, "last", "info.json"))
    if training_info:
        start_step, run_id = training_info
        cfg.training.start_step = start_step  # Set the start step
        cfg.training.run_id = run_id  # Set the run ID
    else:
        cfg.training.start_step = 1  # Set the start step
        cfg.training.run_id = "debug" if isinstance(run, defaultdict) else run["sys"]["id"].fetch()  # Set the run ID

    train(cfg.training, acc, model, ema, dataloaders, optimizer, run)


if __name__ == "__main__":
    cfg_file = OmegaConf.from_cli()['config_file']
    # 输出 PyTorch 版本
    print("PyTorch Version:", torch.__version__)

    # 输出 CUDA 版本
    print("CUDA Version:", torch.version.cuda)

    main(make_config(cfg_file))
