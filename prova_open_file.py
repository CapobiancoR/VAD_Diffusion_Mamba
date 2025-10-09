# Questo file serve unicamente ad aprire le label associate ad un video di testing.
# Questo file esiste some come file di prova per facilitare la comprensione della struttura delle label.

import numpy as np
import cv2
import os

# Percorso del file .npy
file_path = "/media/splash/simone/VAD_Mamba/Avenue_Dataset/test_frame_mask/01_21.npy"

# Carica il file .npy
data = np.load(file_path)

# Mostra i dati caricati
print(data)
print(len(data)); print(sum(data))

# def cut_video(input_path, output_path, start_frame, end_frame):
#     # Apri il video
#     cap = cv2.VideoCapture(input_path)

#     # Ottieni informazioni sul video
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Imposta il codec e il writer per il nuovo video
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     # Vai al frame di inizio
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
#     # Leggi e scrivi i frame richiesti
#     for _ in range(start_frame, end_frame):
#         ret, frame = cap.read()
#         if not ret:
#             break
#         out.write(frame)
    
#     # Rilascia le risorse
#     cap.release()
#     out.release()

# # Esempio di utilizzo
# cut_video("/media/splash/simone/VAD_Mamba/Avenue_Dataset/testing_videos/03.avi",
#            "/home/simone/RaMViD/output101.avi",
#             101, 126)