import os
import torch as th
import torch.nn.functional as F
import numpy as np
import random
import argparse
from diffusion_openai import dist_util, logger
from diffusion_openai.script_util import (
    create_model_and_diffusion,
    args_to_dict,
    model_and_diffusion_defaults,
    add_dict_to_argparser
)
from diffusion_openai.video_datasets import load_data

CHECKPOINT = "/media/splash/simone/VAD_Mamba/log_dicembre/model334000.pt"
SEED = 42

def main():
    args = create_argparser().parse_args()

    # Setup and logging
    dist_util.setup_dist()
    logger.configure("/home/simone/RaMViD/outputs")
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

    # Specifica i gruppi di frame che vuoi processare
    frame_intervals = [(50, 75), (333, 358)]  # Gruppi di frame da analizzare
    seq_len = args.seq_len

    for start_frame, end_frame in frame_intervals:
        logger.log(f"Processing frames from {start_frame} to {end_frame}")

        # Calcola quanti frame includere (end_frame - start_frame)
        num_frames = end_frame - start_frame
        if num_frames != seq_len:
            raise ValueError(f"Intervallo ({start_frame}-{end_frame}) non corrisponde a seq_len ({seq_len})")

        # Load the video dataset per l'intervallo specifico
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
            rgb=args.rgb,
            seq_len=seq_len,
            starting_frame=start_frame
        )

        # Carica il batch
        try:
            video, _ = next(data)
            video = video.to(dist_util.dev())
        except StopIteration:
            logger.log(f"No frames found for interval {start_frame}-{end_frame}")
            continue

        # Condizionamento
        cond_frames = [int(x) for x in args.cond_frames.split(",")]
        cond_kwargs = {"cond_frames": cond_frames, "resampling_steps": args.resample_steps}
        cond_kwargs["cond_img"] = video[:, :, cond_frames].to(dist_util.dev())

        # Sampling
        logger.log("Sampling...")
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        sample = sample_fn(
            model,
            (args.batch_size, 3, seq_len, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            progress=False,
            cond_kwargs=cond_kwargs
        )

        # Compute per-frame MSE
        mse_per_frame = []
        for i in range(seq_len):
            mse = F.mse_loss(sample[:, :, i, :, :], video[:, :, i, :, :])
            mse_per_frame.append(mse.item())

        # Stampa i risultati
        logger.log(f"MSE for frames {start_frame}-{end_frame}: {mse_per_frame}")

    logger.log("Processing complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path=CHECKPOINT,
        seq_len=25,
        cond_frames="0,1,2,3,4,5,10,15,20,24",
        shift_window=5,
        data_dir="",
        image_size=128,
        rgb=True,
        class_cond=False,
        resample_steps=1,
        seed=SEED,
        num_channels=128,
        num_res_blocks=3,
        scale_time_dim=0,
        noise_schedule="linear",
        diffusion_steps=1000,
        cond_generation=True,  # Nuovo argomento
        save_gt=False,         # Nuovo argomento
        frame_intervals="50-75,333-358"  # Nuovo argomento
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # Aggiungi i nuovi argomenti
    #parser.add_argument("--cond_generation", type=bool, default=True, help="Enable conditional generation")
    #parser.add_argument("--save_gt", type=bool, default=False, help="Save ground truth videos")
    #parser.add_argument("--frame_intervals", type=str, default="50-75,333-358",
    #                    help="Comma-separated list of frame intervals in the format start-end")

    return parser



if __name__ == "__main__":
    main()
