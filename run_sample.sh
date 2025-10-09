# #!/bin/bash

# DATA_DIR="/media/splash/simone/VAD_Mamba/Avenue_Dataset/testing_videos/03.avi" # --> per generare un video normale
# # DATA_DIR="/media/nvme_4tb/simone_data/Avenue_Dataset/anomaly_clips/" # --> per fare il qualitativo sull'anomalia

# STARTING_FRAME="0"
# SAVE_DIR="/home/simone/RaMViD/outputs/output_npz"

# MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0"
# DIFFUSION_FLAGS="--diffusion_steps 300 --noise_schedule linear"
# GENERATION_FLAGS="--seq_len 30 \
#     --cond_frames "0,1,10,11,20,21," \
#     --batch_size 1 \
#     --cond_generation True \
#     --num_samples 1 \
#     --resample_steps 1 \
#     --model_path /media/splash/riccardo/log_marzo_300/model004000.pt \
#     --save_gt False\
#     --use_ddim False"

# # Controllo su DATA_DIR
# if [ -f "$DATA_DIR" ]; then
#     # Caso 1: DATA_DIR è un singolo file
#     echo "DATA_DIR is a single file: $DATA_DIR. Proceeding to extract the specified frames..."
#     if [ -n "$STARTING_FRAME" ]; then
#         # Aggiungi lo starting frame se specificato
#         GENERATION_FLAGS="$GENERATION_FLAGS --starting_frame $STARTING_FRAME"
#     fi
# elif [ -d "$DATA_DIR" ]; then
#     # Caso 2: DATA_DIR è una directory
#     echo "DATA_DIR è una directory: $DATA_DIR. Proceeding as usual..."
# else
#     echo "Error: DATA_DIR is not valid."
#     exit 1
# fi

# export PYTHONPATH="/home/simone/RaMViD:$PYTHONPATH"
# echo "PYTHONPATH: $PYTHONPATH"

# python /home/simone/RaMViD/scripts/video_sample.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $GENERATION_FLAGS --save_dir $SAVE_DIR






## Da qui, il codice per adattarlo agli esperimenti:

#!/bin/bash

# Rimuovi (o commenta) i valori fissi:
# DATA_DIR="/media/splash/simone/VAD_Mamba/Avenue_Dataset/testing_videos/"
# STARTING_FRAME="0,25,50,75"
# SAVE_DIR="/home/simone/RaMViD/outputs/output_npz"

# Controllo base: se non viene passato alcun argomento, mostro l'uso e termino
if [ $# -lt 1 ]; then
  echo "Uso: $0 <model_path> [altri_flag...]"
  echo "   <model_path>: percorso alla cartella/file del modello"
  exit 1
fi

# Primo argomento: percorso del modello
MODEL_PATH="$1"


# Quale GPU viene utilizzata
# Funzione che restituisce l'indice della GPU con più memoria libera
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

MODEL_FLAGS="--image_size 128 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0"
DIFFUSION_FLAGS="--diffusion_steps 300 --noise_schedule linear"

# Qui togliamo il --batch_size 4 fisso
# GENERATION_FLAGS="--seq_len 25 \
#    --cond_frames "0,1,12,23,24," \
#    --batch_size 4 \
#    --cond_generation True \
#    --num_samples 1 \
#    --model_path /media/splash/simone/VAD_Mamba/log_dicembre/model334000.pt \
#    --save_gt False"

# Lo riscriviamo SENZA batch_size fisso:

# /media/pinas/riccardo/avenue/model620000.pt \
# /media/pinas/riccardo/STC_check/model460000.pt

GENERATION_FLAGS="--seq_len 30 \
    --cond_frames \"0,1,10,11,20,21,\" \
    --cond_generation True \
    --num_samples 1 \
    --model_path   $MODEL_PATH \
    --save_gt False"


# A questo punto, usiamo i valori di ambiente se esistono
# Altrimenti, se vuoi un default, lo metti con la sintassi ${VAR:-DEFAULT}
DATA_DIR="${DATA_DIR}"
STARTING_FRAME="${STARTING_FRAME}"
BATCH_SIZE="${BATCH_SIZE}"
SAVE_DIR="${SAVE_DIR}"

# Controlli di base (opzionali): se mancano variabili, esci con errore
if [ -z "$DATA_DIR" ]; then
    echo "Errore: manca DATA_DIR"
    exit 1
fi
if [ -z "$STARTING_FRAME" ]; then
    echo "Errore: manca STARTING_FRAME"
    exit 1
fi
if [ -z "$BATCH_SIZE" ]; then
    echo "Errore: manca BATCH_SIZE"
    exit 1
fi
if [ -z "$SAVE_DIR" ]; then
    echo "Errore: manca SAVE_DIR"
    exit 1
fi

# Info di debugging
echo "DATA_DIR: $DATA_DIR"
echo "STARTING_FRAME: $STARTING_FRAME"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "SAVE_DIR: $SAVE_DIR"

export PYTHONPATH="/home/simone/RaMViD:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"


# Ora aggiungiamo questi valori ai parametri di python
python /home/simone/RaMViD/scripts/video_sample.py \
    --data_dir "$DATA_DIR" \
    $MODEL_FLAGS \
    $DIFFUSION_FLAGS \
    $GENERATION_FLAGS \
    --batch_size "$BATCH_SIZE" \
    --starting_frame "$STARTING_FRAME" \
    --save_dir "$SAVE_DIR"
