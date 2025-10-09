#!/bin/bash

# Imposta PYTHONPATH
export PYTHONPATH="/home/simone/RaMViD:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Parametri
DATA_DIR="/media/splash/simone/VAD_Mamba/Avenue_Dataset/testing_videos/03.avi"
MODEL_PATH="/media/splash/simone/VAD_Mamba/log_dicembre/model334000.pt"
IMAGE_SIZE=128
NUM_CHANNELS=128
NUM_RES_BLOCKS=3
SCALE_TIME_DIM=0
DIFFUSION_STEPS=750
NOISE_SCHEDULE="linear"
SEQ_LEN=25
COND_FRAMES="0,1,2,3,4,5,10,15,20,24"
BATCH_SIZE=1
COND_GENERATION=True
NUM_SAMPLES=1
SAVE_GT=False
FRAME_INTERVALS="50-75,333-358"

# Esegui lo script
python /home/simone/RaMViD/scripts/MSE_simone.py \
    --data_dir "$DATA_DIR" \
    --model_path "$MODEL_PATH" \
    --image_size "$IMAGE_SIZE" \
    --num_channels "$NUM_CHANNELS" \
    --num_res_blocks "$NUM_RES_BLOCKS" \
    --scale_time_dim "$SCALE_TIME_DIM" \
    --diffusion_steps "$DIFFUSION_STEPS" \
    --noise_schedule "$NOISE_SCHEDULE" \
    --seq_len "$SEQ_LEN" \
    --cond_frames "$COND_FRAMES" \
    --batch_size "$BATCH_SIZE" \
    --cond_generation "$COND_GENERATION" \
    --num_samples "$NUM_SAMPLES" \
    --save_gt "$SAVE_GT" \
    --frame_intervals "$FRAME_INTERVALS"
