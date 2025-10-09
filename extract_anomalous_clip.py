# In questo file identifichiamo quali porzioni dei video di test sono anomale, così possiamo vedere come il nostro modello gestisce quelle clip.

import cv2
import sys
import os
import numpy as np

controlla_vettore_label= False

# Prova per vedere se i file .npy sono le labels:
labels_path = "/media/nvme_4tb/simone_data/Avenue_Dataset/test_frame_mask/01_02.npy"

# Caricamento del file
if controlla_vettore_label:
    try:
        data = np.load(labels_path)
        print(f"Numero totale di frame: {data.shape[0]}")

        total_sum = np.sum(data)
        print(f"Il numero di frame anomali è: {total_sum}")

        anomalous_indices = np.where(data == 1)[0]
        print(f"I frame anomali sono:\n {anomalous_indices}")

    except Exception as e:
        print(f"Errore nell'apertura del file: {e}")


def extract_anomalous_clip(video_path, labels_path, output_path, clip_length=12):
    # Leggi le label dal file NumPy
    try:
        labels = np.load(labels_path)
        labels = labels.astype(int)
    except Exception as e:
        print(f"Errore nella lettura del file delle label: {e}")
        return

    # Trova il primo frame anomalo
    try:
        start_frame = labels.tolist().index(1) - 2
        start_frame = max(0, start_frame)
    except ValueError:
        print("Non ci sono frame anomali nel file delle label.")
        return

    # Apri il file video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Errore nell'apertura del file video: {video_path}")
        return

    # Imposta il frame di partenza
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Prepara il video in uscita
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Leggi e scrivi i frame
    for _ in range(clip_length):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Rilascia le risorse
    cap.release()
    out.release()
    print(f"Anomalous clip salvato in: {output_path}")

if __name__ == "__main__":
    
    video_path = "/media/nvme_4tb/simone_data/Avenue_Dataset/testing_videos/02.avi"
    labels_path = labels_path
    output_path = "/media/nvme_4tb/simone_data/Avenue_Dataset/anomaly_clips/anomaly_02.avi"
    extract_anomalous_clip(video_path, labels_path, output_path)