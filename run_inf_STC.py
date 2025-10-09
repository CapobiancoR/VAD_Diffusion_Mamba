# In questo file simuliamo la situazione che ci si presenterà durante gli esperimenti.
# Assumeremo di avere un modello che genera "windows" di 25 frame. 
# Partiremo da un video di lunghezza variabile, tra i 40 e i 1500 frames circa. 
# Simuleremo le batch da 4/5 windows alla volta e vedremo come gestire gli estremi.

# ---> Una volta capito come far funzionare il tutto, questo script può essere eliminato <---
# import math

# window_size = 25
# video_len = 201  # Prova con vari valori: 100, 101, 120, ecc.

# # 1) Calcolo del numero di finestre "ingenue"
# batch_size = math.ceil(video_len / window_size)

# # 2) Calcolo di quanto sforiamo rispetto alla lunghezza del video
# leftover = (batch_size * window_size) - video_len; print(f"Leftover is: {leftover}\n")

# current_frame = 0

# for i in range(batch_size):
#     start_frame = current_frame
#     end_frame = current_frame + window_size - 1

#     # 3) Se siamo all'ultima finestra e c'è da "recuperare" leftover,
#     #    spostiamo l'inizio finestra all'indietro.
#     if i == batch_size - 1 and leftover > 0:
#         start_frame -= leftover
#         end_frame = start_frame + window_size - 1

#     print(f"La finestra numero {i} copre i frame ({start_frame}, {end_frame})")

#     # Aggiorniamo il frame di partenza per la finestra successiva
#     current_frame += window_size

import os
from pathlib import Path
import wandb
from typing import Optional
import os
import cv2
import math
import subprocess
import numpy as np
import datetime
from visualize_video import save_video_from_npz

# def compute_start_frames_from_video(video_path, window_size=25, max_batch=4): # --> funzione vecchia che funzionava con stride = N
#     """
#     Calcola la lunghezza del video e le finestre da generare.
#     Ritorna:
#         - sf_list: lista di stringhe con gli starting frames (es: ["1,26,51,76", ...])
#         - bs_list: lista dei batch size (es: [4,4,4])
#         - video_len: numero di frame totali del video
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Impossibile aprire il video: {video_path}")
#     video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()

#     n_windows = math.ceil(video_len / window_size)
#     leftover = (n_windows * window_size) - video_len

#     start_frames = []
#     current_start = 0
#     for i in range(n_windows):
#         if i == n_windows - 1 and leftover > 0:
#             current_start -= leftover
#         start_frames.append(current_start)
#         current_start += window_size

#     # Raggruppa in batch di max_batch
#     chunks = [start_frames[i:i+max_batch] for i in range(0, len(start_frames), max_batch)]
#     sf_list = []
#     bs_list = []

#     for chunk in chunks:
#         sf_list.append(",".join(str(s) for s in chunk))
#         bs_list.append(len(chunk))

#     return sf_list, bs_list, video_len


def compute_start_frames_from_video(video_path, window_size=30, stride=15, max_batch=5):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossibile aprire il video: {video_path}")
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap.release()

    start_frames = []
    current_start = 0
    while current_start + window_size <= video_len: # @riccardo , penso questo fosse sbagliato current_start + window_size <= video_len:
        start_frames.append(current_start)
        current_start += stride

    # Assicurati di includere l'ultima clip che copre l'ultimo frame
    if video_len >= window_size:
        last_start = video_len - window_size
        # Aggiungi last_start anche se è già presente per garantire che
        # l'ultima clip termini esattamente all'ultimo frame
        start_frames.append(last_start)

    # Raggruppa gli starting frames in batch di dimensione massima max_batch
    chunks = [start_frames[i:i+max_batch] for i in range(0, len(start_frames), max_batch)]
    sf_list = []
    bs_list = []
    for chunk in chunks:
        sf_list.append(",".join(str(s) for s in chunk))
        bs_list.append(len(chunk))

    return sf_list, bs_list, video_len

def get_latest_model_file(directory: str) -> Optional[str]:
    """
    Cerca nella cartella `directory` tutti i file che hanno "model" nel filename
    (case-insensitive) e restituisce il percorso del file più "nuovo" (in base
    a ctime, cioè time of creation su Windows, o time of last metadata change su Unix).
    Se non viene trovato nessun file corrispondente, restituisce None.

    Args:
        directory (str): Path della cartella da scandire.

    Returns:
        Optional[str]: Percorso (stringa) del file contenente "model" nel nome con
                       data di creazione più recente, o None se nessun file corrisponde.
    """
    # Converto in Path per comodità
    folder = Path(directory)
    if not folder.is_dir():
        raise ValueError(f"Il percorso {directory!r} non è una cartella valida.")

    latest_file: Optional[Path] = None
    latest_ctime: float = 0.0

    # Itero su tutti i file all'interno (non scende nelle sottocartelle).
    for file_path in folder.iterdir():
        # Verifico che sia un file (non una cartella) e che "model" compaia nel nome
        if file_path.is_file() and "model" in file_path.name.lower():
            # Prendo il tempo di creazione (o metadata change su Unix)
            try:
                ctime = file_path.stat().st_ctime
            except OSError:
                # Se per qualche motivo non riesco a leggere lo stat, salto questo file
                continue

            # Confronto con il più recente finora trovato
            if latest_file is None or ctime > latest_ctime:
                latest_file = file_path
                latest_ctime = ctime

    # Se non è stato trovato alcun file, ritorno None, altrimenti la stringa del path
    return str(latest_file) if latest_file is not None else None


if __name__ == "__main__":
    
    

    folder_videos ="/home/riccardo/STC/shanghaitech/testing/frames"  # "/media/pinas/riccardo/UBnormal_video/UB_test/"#"/home/riccardo/Avenue_Dataset/testing_videos/" #   
   
    
    checkpoint_folder = "/home/riccardo/checkpoints_STC"
    
    last_model = get_latest_model_file(checkpoint_folder)
    steps = last_model.split('/')[-1].split('_')[-1].split(".")[0] if last_model else "unknown"
    
    base_output_folder = f"/media/pinas/riccardo/OUT_STC/STC_outputs_{steps}train_200step" #"/media/pinas/riccardo/outputs_620train_200step"  #  Cartella dove creare le sottocartelle
    
    bash_script = f"/home/riccardo/RaMViD/run_sample.sh {last_model}"  # Path al tuo script bash, con il placeholder per il modello
    window_size = 30
    stride = 15
    max_batch = 5 
    
    os.makedirs(base_output_folder, exist_ok=True)  # Crea la cartella di output se non esiste
    
    wandb.init(project='RaMViD_', entity='pinlab-sapienza',
                   config={
                "checkpoint_folder": checkpoint_folder,
                "last_model": last_model,
                "steps": steps,
                "window_size": window_size,
                "stride": stride,
                "max_batch": max_batch,
                "base_output_folder": base_output_folder,
                "bash_script": bash_script,
                "folder_videos": folder_videos
            },
            name='INF_RaMViD_'+checkpoint_folder+"_"+steps
        )

    for file_name in os.listdir(folder_videos):
        # if file_name != "18.avi": # --> se vuoi fermarti ad un solo video specifico
        #     continue

        if not file_name.endswith(".mp4") and not file_name.endswith(".avi"):
            print(f"File '{file_name}' non è un video supportato (deve essere .mp4 o .avi): salto.")
            continue

        video_name = os.path.splitext(file_name)[0]
        video_out_folder = os.path.join(base_output_folder, video_name)

        if os.path.isdir(video_out_folder) and len(os.listdir(video_out_folder)) > 0:
            print(f"Cartella '{video_out_folder}' già esistente con file presenti: salto il video '{file_name}'.")
            continue

        # Trovato "video.avi"
        video_path = os.path.join(folder_videos, file_name)

        # 1) Calcola gli intervalli
        sf_list, bs_list, vlen = compute_start_frames_from_video(video_path, window_size, stride, max_batch)
        print(f"\n>>> Video selezionato: {file_name} ({vlen} frames)")
        print(f"    Starting frames list: {sf_list}")
        print(f"    Batch size list:     {bs_list}")
        
        log_path = os.path.join(video_path.split('.')[0], "log.txt")
        os.makedirs(video_path.split('.')[0], exist_ok=True)  # Crea la cartella se non esiste
        with open(log_path, "a") as log_file:
            log_file.write(f"\n>>> Video selezionato: {file_name} ({vlen} frames)")

        # 2) Crea la cartella col nome del video, ad esempio "18" (senza estensione .avi)
        video_name = os.path.splitext(file_name)[0]  # Es: "18"
        video_out_folder = os.path.join(base_output_folder, video_name)  # Es: "/home/simone/RaMViD/outputs/output_npz/18"
        os.makedirs(video_out_folder, exist_ok=True)  # Crea la cartella se non esiste
        print(f"Cartella di output: {video_out_folder}")

        # 3) Esegui i chunk uno per uno
        for idx, (sf_string, b_size) in enumerate(zip(sf_list, bs_list)):
            print(f"\n    [Generazione {idx+1}/{len(sf_list)}] -> STARTING_FRAME={sf_string}, BATCH_SIZE={b_size}") 

            # Costruisci il comando per eseguire il Bash:
            #   - DATA_DIR=...
            #   - STARTING_FRAME=...
            #   - BATCH_SIZE=...
            #   - SAVE_DIR=video_out_folder
            command_str = (
                f"DATA_DIR='{video_path}' "
                f"STARTING_FRAME='{sf_string}' "
                f"BATCH_SIZE='{b_size}' "
                f"SAVE_DIR='{video_out_folder}' "
                f"bash {bash_script}"
            )
            
            # 4) Avvia il bash script (unisci in stringa per shell=True)
            result = subprocess.run(command_str, shell=True)
            if result.returncode != 0:
                print("    ERRORE nella generazione. Interrompo.")
                break
            
            # Se il tuo script `video_sample.py` salva già i file .npz dentro `video_out_folder`,
            # con un nome unico (es. "128x128x25.npz", o un shape simile),
            # NON serve rinominare/spostare nulla. Avrai i chunk subito in "18/".
            
            # Se invece video_sample.py salva *sempre* con un nome fisso (es: "generated.npz")
            # allora potresti doverlo rinominare manualmente.
            # Puoi commentare o togliere del tutto la parte sottostante se non ti serve.

            
            # # ESEMPIO di rename manuale (se necessario): ---> QESTO FUNZIONAVA CON STRIDE = N
            #  generated_path = os.path.join(video_out_folder, "beginning.npz")
            # if os.path.exists(generated_path):
            #     new_name = f"{video_name}_chunk{idx}.npz"
            #     out_path = os.path.join(video_out_folder, new_name)
            #     os.rename(generated_path, out_path)
            #     print(f"    Salvataggio chunk: {out_path}")
            # else:
            #     print("    ATTENZIONE: Non trovo il file .npz generato!")  
            generated_path = os.path.join(video_out_folder, "beginning.npz")

            if os.path.exists(generated_path):
                now = datetime.datetime.now()
                time_str = now.strftime("%H%M%S")    # Orario nel formato HHMMSS

                # Pulisce sf_string: rimuove la virgola finale e sostituisce le virgole con un trattino
                sf_clean = sf_string.rstrip(",").replace(",", "-")

                # Costruisce il nuovo nome, ad es.: "10-15-20-25-30_173512_5x30x128x128x3.npz"
                new_name = f"{sf_clean}_{time_str}_{b_size}x{window_size}.npz"
                out_path = os.path.join(video_out_folder, new_name)

                os.rename(generated_path, out_path)
                print(f"Salvataggio chunk: {out_path}")
            else:
                print("ATTENZIONE: Non trovo il file .npz generato!")
                
    # CREO VIDEO AVI A PARTIRE DA NPZ GENERATI
    
    input_dir = base_output_folder#"/media/pinas/riccardo/UB_outputs_460train_200step/" #"/home/riccardo/RaMViD/outputs_1000/"
    output_dir = base_output_folder+"/avi/" #"/media/pinas/riccardo/UB_outputs_460train_200step/avi/" #"/home/riccardo/RaMViD/outputs_1000/avenue_avi/"
    os.makedirs(output_dir, exist_ok=True)
    
    for video_number in os.listdir(input_dir):

        path=input_dir+video_number
        #os.makedirs(path+"/avi_aggregated_format", exist_ok=True)

        # Esempio di utilizzo:
        save_video_from_npz(
                generated_path=path,
                output_file=output_dir+video_number+".avi",
                fps=30,
                upsample_size=(856, 480) #Avenue 640x360 STC 856,480  UB 1080,720 # Imposta la dimensione di upsample desiderata
            )
    