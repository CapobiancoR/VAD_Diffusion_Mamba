import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import random
import os, sys
sys.path.insert(1, os.getcwd())

from diffusion_openai.video_datasets import load_data
from diffusion_openai import dist_util, logger
from diffusion_openai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

CHECKPOINT = "/media/splash/simone/VAD_Mamba/log_dicembre/model334000.pt"

def main():
    args = create_argparser().parse_args()

    # Setup and logging
    dist_util.setup_dist()
    logger.configure("/home/lucaromani/RaMViD/outputs")
    if args.seed:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Model and diffusion setup
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    # Load the specified clip from dataset
    logger.log("Loading data...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        rgb=args.rgb,
        seq_len=args.seq_len
    )


    channels = 3 if args.rgb else 1
    all_mse = []

    # Load the video clip once before the loop
    video, _ = next(data)
    video = video.to(dist_util.dev())  # Move to device

    # Iterate through the shifts
    for start in range(0, args.seq_len - args.cond_frames, args.shift_window):
        # Set conditioning frames
        cond_frames = list(range(start, start + args.cond_frames))
        ref_frames = list(i for i in range(args.seq_len) if i not in cond_frames)
        logger.log(f"Cond_frames: {cond_frames}")
        
        # Prepare conditioning arguments
        cond_kwargs = {"cond_frames": cond_frames, "resampling_steps": args.resample_steps}
        cond_kwargs["cond_img"] = video[:, :, cond_frames].to(dist_util.dev())
        
        # Sampling
        logger.log("Sampling...")
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, channels, args.seq_len, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            progress=False,
            cond_kwargs=cond_kwargs
        )

        print(f"Sample shape: {sample.shape}")
        print(f"Video shape: {video.shape}")

        # Compute per-frame MSE
        mse_per_frame = []
        for i in range(args.seq_len):
            mse = F.mse_loss(sample[:, :, i, :, :], video[:, :, i, :, :])
            mse_per_frame.append(mse.item())
        
        logger.log(f"MSE per frame for window starting at {start}: {mse_per_frame}")
        all_mse.append(mse_per_frame)


        # Output stacked MSE results
        logger.log(f"Stacked MSE values per shift: {all_mse}")
        np.savez(os.path.join(logger.get_dir(), "mse_per_frame_shifts.npz"), all_mse=all_mse)
        dist.barrier()
        logger.log("Processing complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        seq_len=50,
        cond_frames=10,
        shift_window=5,
        num_iterations=5,
        cond_generation=True,
        resample_steps=1,
        data_dir='',
        save_gt=True,
        seed=0,
        specific_clip_index=0,  # Index of the clip to use as example
        rgb=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    import time
    start = time.time()
    main()
    end = time.time()
    print(f"Elapsed time: {end - start}")