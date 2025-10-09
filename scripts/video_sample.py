"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse

import numpy as np
import torch as th
import torch.distributed as dist

import os, sys
sys.path.insert(1, os.getcwd()) 
import random
import time

from diffusion_openai.video_datasets import load_data
from diffusion_openai import dist_util, logger
from diffusion_openai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    
    print(f"Available GPUs: {th.cuda.device_count()}")
    #for i in range(th.cuda.device_count()):
    #    print(f"GPU {i}: {th.cuda.get_device_name(i)}, Memory Allocated: {th.cuda.memory_allocated(i)/1024**2:.2f} MB")
    #th.cuda.set_device(2)
    print(f"Current device: {th.cuda.current_device()}")

    start_total = time.time() #@simone_tempo

    args = create_argparser().parse_args()

    #@riccardo test per problema memoria ad inferenza
    #dist_util.setup_dist()
    logger.configure()
    if args.seed:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # @simone Aggiunto per poter specificare la directory
    save_dir = args.save_dir if args.save_dir else logger.get_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)  # Crea la directory se non esiste
    logger.log(f"Samples will be saved to: {save_dir}")    

    logger.log("creating model and diffusion...")
    model_start = time.time() #@simone_tempo
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    logger.log(f"XXX Tempo per creare il modello e la diffusione: {time.time() - model_start:.2f} secondi") #@simone_tempo

    start = time.time() #@simone_tempo
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    logger.log(f"XXX Tempo per caricare il modello e spostarlo su device: {time.time() - start:.2f} secondi") #@simone_tempo

    cond_kwargs = {}
    cond_frames = []

    # @simone
    if os.path.isdir(args.data_dir):
        # If we pass the entire directory, we don't care about the starting_frame
        starting_frame = None
    else:
        # If we pass a specific file, we might be interestes in conditioning the specific starting frames
        if args.starting_frame:
            if "," in args.starting_frame:
                # multiple frames (eg: "0,101,200,301"). This is useful for batch generation
                starting_frame = [
                    int(x) for x in args.starting_frame.split(",") if x.strip() != ""
                ]
            else:
                # single integer, eg: "101"
                starting_frame = [int(args.starting_frame)]
            logger.log(f"Using starting_frame: {starting_frame}")
        else:
            starting_frame = None
        
    if args.cond_generation:
        start = time.time() #@simone_tempo
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
            rgb=args.rgb,
            seq_len=args.seq_len,
            starting_frame=starting_frame  # @simone
        )

        string_cond = args.cond_frames.strip().strip('"').rstrip(',')
        cond_frames = [int(x) for x in string_cond.split(",") if x.strip() != ""]
        print(f"cond_frames: {cond_frames}")

        ref_frames = [i for i in range(args.seq_len) if i not in cond_frames]
        logger.log(f"ref_frames: {ref_frames}")

        cond_kwargs = {"cond_frames": cond_frames, "resampling_steps": args.resample_steps}

        #logger.log(f"cond_frames: {cond_frames}")
        #logger.log(f"ref_frames: {ref_frames}")
        logger.log(f"seq_len: {args.seq_len}")
        cond_kwargs["resampling_steps"] = args.resample_steps
        logger.log(f"XXX Tempo per il caricamento dei dati: {time.time() - start:.2f} secondi") #@simone_tempo

    cond_kwargs["cond_frames"] = cond_frames

    if args.rgb:
        channels = 3
    else:
        channels = 1

    logger.log("sampling...")
    sampling_start = time.time() #@simone_tempo

    all_videos = []
    all_gt = []
    while len(all_videos) * args.batch_size < args.num_samples:
        start = time.time() #@simone_tempo
        
        if args.cond_generation:
            video, _ = next(data)
            cond_kwargs["cond_img"] = video[:,:,cond_frames].to(dist_util.dev()) 
            video = video.to(dist_util.dev())


        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (args.batch_size, channels, args.seq_len, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            progress=True, #@Riccardo show progress bar
            cond_kwargs=cond_kwargs
        )    

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 4, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_videos.extend([sample.cpu().numpy() for sample in gathered_samples])
        logger.log(f"created {len(all_videos) * args.batch_size} samples")
        logger.log(f"XXX Tempo per il batch di sampling: {time.time() - start:.2f} secondi") #@simone_tempo

        if args.cond_generation and args.save_gt:

            video = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8)
            video = video.permute(0, 2, 3, 4, 1)
            video = video.contiguous()

            gathered_videos = [th.zeros_like(video) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_videos, video)  # gather not supported with NCCL
            all_gt.extend([video.cpu().numpy() for video in gathered_videos])
            logger.log(f"created {len(all_gt) * args.batch_size} videos")

    logger.log(f"XXX Tempo totale di sampling: {time.time() - sampling_start:.2f} secondi") #@simone_tempo
    arr = np.concatenate(all_videos, axis=0)

    if args.cond_generation and args.save_gt:
        arr_gt = np.concatenate(all_gt, axis=0)


    # if dist.get_rank() == 0: --> Questo funzionava per video di 25 frames
    #     # @simone modificato per poter gestire il salvataggio nella directory specificata
    #     timestamp = time.strftime("%H_%M_%S") # @simone questo l'ho messo per personalizzare meglio il nome del file
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     filename = f"{timestamp}_{shape_str}.npz"
    #     output_path = os.path.join(save_dir, filename)
    #     logger.log(f"saving samples to {output_path}")
    #     np.savez(output_path, arr)

    #     if args.cond_generation and args.save_gt:
    #         shape_str_gt = "x".join([str(x) for x in arr_gt.shape])
    #         filename_gt = f"{timestamp}_{shape_str_gt}.npz"
    #         output_path_gt = os.path.join(save_dir, filename_gt)
    #         logger.log(f"saving ground_truth to {output_path_gt}")
    #         np.savez(output_path_gt, arr_gt)

    if dist.get_rank() == 0:
        # Costruzione del nome del file standard
        timestamp = time.strftime("%H_%M_%S")  # es. "17_35_10"
        shape_str = "x".join([str(x) for x in arr.shape])  # es. "5x30x128x128x3"
        filename = f"{timestamp}_{shape_str}.npz"
        output_path = os.path.join(save_dir, filename)
        
        logger.log(f"saving samples to {output_path}")
        np.savez(output_path, arr)
        
        # Scrittura/aggiornamento del file di log (log.txt) per associare il file salvato agli starting frame
        log_path = os.path.join(save_dir, "log.txt")
        
        # Recupera lo starting frame passato come argomento.
        # Se args.starting_frame è una stringa tipo "10,15,20,25,30," la lasciamo com'è (o la puliamo)
        if args.starting_frame:
            sf_clean = args.starting_frame.rstrip(",")
        else:
            sf_clean = "none"
        
        # Apri il log in modalità append e scrivi la riga
        with open(log_path, "a") as log_file:
            log_file.write(f"{filename}: starting_frame = {sf_clean}\n")
        
        logger.log(f"Log aggiornato: {log_path}")
        
        if args.cond_generation and args.save_gt:
            shape_str_gt = "x".join([str(x) for x in arr_gt.shape])
            filename_gt = f"{timestamp}_{shape_str_gt}.npz"
            output_path_gt = os.path.join(save_dir, filename_gt)
            logger.log(f"saving ground_truth to {output_path_gt}")
            np.savez(output_path_gt, arr_gt)

    dist.barrier()
    logger.log("sampling complete")
    end_total = time.time() #@simone_tempo
    logger.log(f"Tempo totale di esecuzione: {end_total - start_total:.2f} secondi") #@simone_tempo


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1, # @simone: qui era settato a 10
        batch_size=1, # @simone: qui era settato a 10
        use_ddim=False,
        model_path="",
        seq_len=12,
        sampling_type="generation",
        cond_frames="0,",
        cond_generation=True,
        resample_steps=1,
        data_dir="",
        save_gt=False,
        seed = 0,
        starting_frame = "", # @simone
        save_dir="" # @simone
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
    print(f"elapsed time: {end - start}")