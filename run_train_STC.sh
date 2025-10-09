#!/bin/bash



MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0" #image_size 224 num_channels 128 num_res_blocks 3
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 2e-5 --batch_size 1 --microbatch 1 --seq_len 30 --max_num_mask_frames 24 --uncondition_rate 0.25" #lr 2e-5 batch_size 8 microbatch 2 seq_len 6 max_num_mask_frames 4 uncondition_rate 0.25

# NOTA: nelle nostre prove funzionanti, usavamo batch_size=2 e microbatch=2

# Trova i file (-maxdepth 1) contenenti "model" nel nome,
# stampa tempo di modifica + percorso, ordina decrescente,
# prende il primo e isola il percorso

###  VARIABILI DA MODIFICARE PER CAMBIO DATASET ###

DIR="/home/riccardo/checkpoints_STC" #"/media/pinas/riccardo/UB_check" #"/media/pinas/riccardo/avenue" #"/media/pinas/riccardo/STC_check"
DATA_DIR="/home/riccardo/STC/shanghaitech/training/videos" #"/home/riccardo/UB_training" #"/home/riccardo/STC/shanghaitech/training/videos" #  "/home/riccardo/Avenue_Dataset/training_videos"  # 

export DIFFUSION_BLOB_LOGDIR="/home/riccardo/checkpoints_STC" #"/media/pinas/riccardo/UB_check" #"/media/pinas/riccardo/avenue" 

latest_file=$(find "$DIR" -maxdepth 1 -type f -name '*model*' -printf '%T@ %p\n' \
  | sort -nr \
  | head -n1 \
  | cut -d' ' -f2-)

if [[ -z "$latest_file" ]]; then
  echo "Nessun file con 'model' trovato in '$DIR'." >&2
  exit 1
fi

echo "File più recente con 'model' nel nome:"
echo "$latest_file"




RESUME_CHECKPOINT="--resume_checkpoint  $latest_file" #/media/pinas/riccardo/UB_check/UB_training_model456000.pt" #/home/riccardo/checkpoints_STC/" #  /home/riccardo/checkpoints_ub/model448000.pt" #/media/pinas/riccardo/avenue/model488000.pt" #

get_free_gpu_by_mem() {
  # Estrae righe del tipo "0, 12345" dove 0 è l'indice e 12345 è la memoria libera (in MiB)
  # Ordina per secondo campo (memoria libera) in ordine decrescente
  # Prende il primo elemento e ne estrae l'indice
  local gpu_idx
  gpu_idx=$(
    nvidia-smi --query-gpu=index,memory.free \
               --format=csv,noheader,nounits 2>/dev/null | \
    sort -t',' -k2 -nr | \
    head -n1 | \
    awk -F',' '{print $1}'
  )
  echo "$gpu_idx"
}

# Chiamata alla funzione
GPU=$(get_free_gpu_by_mem)
if [ -z "$GPU" ]; then
  echo "Impossibile rilevare una GPU libera (nvidia-smi ha restituito errore o non ci sono GPU)."
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU
echo "Attenzione: sto usando la GPU $GPU (memoria libera massima)."
echo "Utilizzo della GPU: $CUDA_VISIBLE_DEVICES"

echo $DATA_DIR

python scripts/video_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $RESUME_CHECKPOINT
