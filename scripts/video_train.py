"""
Train a diffusion model on images.
"""

import argparse
import os, sys
sys.path.insert(1, os.getcwd()) 
import torch.distributed as dist
import numpy as np

import torch as th
from diffusion_openai import dist_util, logger
from diffusion_openai.video_datasets import load_data
from diffusion_openai.resample import create_named_schedule_sampler
from diffusion_openai.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion_openai.train_util import TrainLoop

import time
import threading
import psutil
import logging

# @simone questi controlli li ho messi io:
print(f"Available GPUs: {th.cuda.device_count()}")
for i in range(th.cuda.device_count()):
    print(f"GPU {i}: {th.cuda.get_device_name(i)}, Memory Allocated: {th.cuda.memory_allocated(i)/1024**2:.2f} MB")
#th.cuda.set_device(3)
print(f"Current device: {th.cuda.current_device()}")

########################### MONITORAGGIO CPU, GPU, VRAM ###########################
# Proviamo a usare pynvml per monitorare la GPU
try:
    import pynvml
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except ImportError:
    gpu_handle = None

# Configuriamo il logging su file
logging.basicConfig(
    filename='/home/simone/RaMViD/resource_log.txt', 
    level=logging.INFO,
    format='%(asctime)s: %(message)s'
)

# Variabile globale da aggiornare per indicare la fase corrente del training
current_stage = "Initialize training loop"

def log_resources(stage_info):
    cpu_percent = psutil.cpu_percent()
    ram_percent = psutil.virtual_memory().percent
    gpu_mem_percent = 0
    gpu_util = 0
    if th.cuda.is_available() and gpu_handle is not None:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_mem_percent = mem_info.used / mem_info.total * 100
        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
        gpu_util = util.gpu
    logging.info(
        f"{stage_info} | CPU: {cpu_percent}% | RAM: {ram_percent}% | "
        f"GPU Memory: {gpu_mem_percent:.2f}% | GPU Utilization: {gpu_util}%"
    )

def monitor_resources():
    while True:
        log_resources(current_stage)
        time.sleep(10)

# Avvio del thread di monitoraggio in background
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()
#################################################################################


def main():

    global current_stage # @simone monitoring cpu&gpu
    parser, defaults = create_argparser()
    args = parser.parse_args()
    parameters = args_to_dict(args, defaults.keys())
    # th.manual_seed(args.seed)
    # np.random.seed(args.seed)

    #dist.init_process_group(backend="nccl") #@Riccardo test multi gpu
    dist_util.setup_dist()
    logger.configure()
    for key, item in parameters.items():
        logger.logkv(key, item)
    logger.dumpkvs()

    current_stage = "Creating model and diffusion" # @simone monitor cpu&gpu
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    print(dist_util.dev())
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    current_stage = "Creating dataloader" # @simone monitor cpu&gpu
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=False,
        rgb=args.rgb,
        seq_len=args.seq_len
    )

    if args.mask_range is None:
        mask_range = [0, args.seq_len]
    else:
        mask_range = [int(i) for i in args.mask_range if i != ","]
    
    current_stage = "Start training" # @simone monitor cpu&gpu
    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        clip=args.clip,
        anneal_type=args.anneal_type,
        steps_drop=args.steps_drop,
        drop=args.drop,
        decay=args.decay,
        max_num_mask_frames=args.max_num_mask_frames,
        mask_range=mask_range, 
        uncondition_rate=args.uncondition_rate,
        exclude_conditional=args.exclude_conditional,
        
        #riccardo custom args
        wandb_name = (args.data_dir,args.resume_checkpoint,args.diffusion_steps)
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=64,
        microbatch=32,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip=1,
        seed=123,
        anneal_type=None,
        steps_drop=0.0,
        drop=0.0,
        decay=0.0,
        seq_len=25, # @simone NUMERO DI FRAME
        max_num_mask_frames=4,
        mask_range=None,
        uncondition_rate=0.0,
        exclude_conditional=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser, defaults


if __name__ == "__main__":
    main()