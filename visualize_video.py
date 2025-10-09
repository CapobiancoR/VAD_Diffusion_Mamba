import numpy as np
import cv2
import os
import math

import numpy as np
import cv2
import os

def save_video_from_npz(generated_path, output_file="output.avi", fps=25, upsample_size=(1080,720)):  #Avenue 640x360 STC 856,480  UB 1080,720
    """
    Carica i file .npz da 'generated_path' (se è una cartella, li unisce lungo la dimensione batch)
    e salva un singolo file AVI in cui i frame sono ottenuti appiattendo l'array (B_total, T, H, W, C)
    in una sequenza di (B_total*T) frame. Prima di salvarli, ogni frame viene upsamplato a 'upsample_size'.

    Ad esempio:
      Se hai file di forma:
         5x25x128x128, 5x25x128x128, 2x25x128x128
      otterrai un array finale di forma 12x25x128x128,
      che verrà trasformato in 300 frame (12*25) da salvare nel video,
      dove ogni frame viene ridimensionato a upsample_size (640x360).
    """
    # Carica e unisci i file .npz
    if os.path.isdir(generated_path):
        # Prendi tutti i .npz (escludendo eventuali file _gt.npz)
        npz_files = [os.path.join(generated_path, f) for f in os.listdir(generated_path)
                     if f.endswith(".npz") and "_gt" not in f]
        if not npz_files:
            print("Nessun file .npz trovato in", generated_path)
            return

        # Ordina i file per data di creazione
        npz_files = sorted(npz_files, key=lambda f: os.path.getctime(f))
        print("File .npz trovati:", npz_files)
        videos = []
        for f in npz_files:
            data = np.load(f, allow_pickle=True) #@riccardo aggiunto allow_pickle=True 
            key = list(data.keys())[0]
            video = data[key]  # atteso formato: (B, T, H, W) oppure (B, T, H, W, C)
            videos.append(video)
        
        # Controlla che T, H e W siano uguali tra i file
        T, H, W = None, None, None
        for vid in videos:
            if T is None:
                T = vid.shape[1]
                H = vid.shape[2]
                W = vid.shape[3]
            else:
                if vid.shape[1] != T or vid.shape[2] != H or vid.shape[3] != W:
                    print("I file hanno dimensioni differenti in T, H o W!")
                    return
        
        # Concatena lungo la dimensione batch (axis=0)
        final_video = np.concatenate(videos, axis=0)  # forma: (B_total, T, H, W) o (B_total, T, H, W, C)
    else:
        data = np.load(generated_path,allow_pickle=True) #@riccardo aggiunto allow_pickle=True
        key = list(data.keys())[0]
        final_video = data[key]
        T = final_video.shape[1]
        H = final_video.shape[2]
        W = final_video.shape[3]
    
    # Se il video è in formato 4D (B, T, H, W) => aggiungi il canale e replicalo per ottenere RGB
    if final_video.ndim == 4:
        final_video = np.expand_dims(final_video, axis=-1)  # Diventa (B, T, H, W, 1)
        final_video = np.repeat(final_video, 3, axis=-1)      # Ora (B, T, H, W, 3)
    elif final_video.ndim == 5:
        if final_video.shape[-1] == 1:
            final_video = np.repeat(final_video, 3, axis=-1)
        elif final_video.shape[-1] != 3:
            print("Il video ha un numero di canali non supportato:", final_video.shape[-1])
            return
    else:
        print("Formato video non supportato.")
        return

    B_total, T, H, W, C = final_video.shape
    print(f"Video finale: {B_total} video, {T} frame ciascuno, dimensione frame originale: {H}x{W}")

    # Appiattisci le dimensioni batch e tempo per ottenere (B_total*T, H, W, C)
    frames = final_video.reshape((-1, H, W, C))
    total_frames = frames.shape[0]
    print(f"Numero totale di frame da salvare: {total_frames}")

    # Imposta la dimensione di upsample (width, height)
    upsample_width, upsample_height = upsample_size

    # Configura il VideoWriter: ogni frame sarà upsamplato a (upsample_width, upsample_height)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_file, fourcc, fps, (upsample_width, upsample_height))

    # Scrivi ogni frame nel video, upsamplando a 640x360
    for idx, frame in enumerate(frames):
        frame_uint8 = frame.astype(np.uint8)
        # Converti da RGB a BGR per cv2
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        # Upsample il frame a upsample_size
        frame_resized = cv2.resize(frame_bgr, (upsample_width, upsample_height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame_resized)
    
    writer.release()
    print("Video salvato in", output_file)



if __name__ == "__main__":
    
    input_dir = "/media/pinas/riccardo/OUT_UB/UB_outputs_model808000train_200step/" #"/home/riccardo/RaMViD/outputs_1000/"
    output_dir = "/media/pinas/riccardo/OUT_UB/UB_outputs_model808000train_200step/avi/" #"/home/riccardo/RaMViD/outputs_1000/avenue_avi/"
    os.makedirs(output_dir, exist_ok=True)
    
    for video_number in os.listdir(input_dir):

        path=input_dir+video_number
        #os.makedirs(path+"/avi_aggregated_format", exist_ok=True)

        # Esempio di utilizzo:
        save_video_from_npz(
                generated_path=path,
                output_file=output_dir+video_number+".avi",
                fps=25
            )