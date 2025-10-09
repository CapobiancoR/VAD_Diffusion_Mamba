# Inizialmente questo codice serviva solo a dimostrare che si può "squashare" e "de-squashare" un video in maniera deterministica
# con gli strumenti di pytorch. Poi lo abbiamo usato per prendere gli output del nostro modello e riportarli nella dimensione originale.


### VIDEO
import cv2
import torch
import torch.nn.functional as F
import numpy as np

voglio_solo_una_demo = False

if voglio_solo_una_demo:
    # Percorso video input e output
    video_path = "/home/simone/RaMViD/video_prova.mp4"
    squashed_video_path = "/home/simone/RaMViD/video_squashed.mp4"
    restored_video_path = "/home/simone/RaMViD/video_restored.mp4"

    # Parametri di ridimensionamento
    target_size = (128, 128)  # Nuova risoluzione

    # Caricamento video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Writer per video di output
    out_squashed = cv2.VideoWriter(squashed_video_path, fourcc, fps, target_size)
    out_restored = cv2.VideoWriter(restored_video_path, fourcc, fps, (original_width, original_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converti il frame da BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Converti frame in tensor PyTorch
        frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Ridimensiona a risoluzione target
        squashed_tensor = F.interpolate(frame_tensor, size=target_size, mode="bilinear")

        # Ripristina la risoluzione originale
        restored_tensor = F.interpolate(squashed_tensor, size=(original_height, original_width), mode="bilinear")

        # Converti tensor in immagine
        squashed_frame = (squashed_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype("uint8")
        restored_frame = (restored_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype("uint8")

        # Converti da RGB a BGR per OpenCV
        squashed_frame_bgr = cv2.cvtColor(squashed_frame, cv2.COLOR_RGB2BGR)
        restored_frame_bgr = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR)

        # Scrivi i frame nei rispettivi video
        out_squashed.write(squashed_frame_bgr)
        out_restored.write(restored_frame_bgr)

    cap.release()
    out_squashed.release()
    out_restored.release()

    print(f"Video schiacciato salvato in: {squashed_video_path}")
    print(f"Video ripristinato salvato in: {restored_video_path}")

#########
import cv2
import torch
import torch.nn.functional as F

# Percorso video input e output
squashed_video_path = "/home/simone/RaMViD/outputs/outputs25/video_0.avi"
restored_video_path = "/home/simone/RaMViD/outputs/outputs25/video_0_RESTORED.avi"

# Risoluzione originale (da impostare manualmente o derivare dal video originale)
original_width = 640  # Sostituisci con la larghezza originale
original_height = 360  # Sostituisci con l'altezza originale

# Caricamento video squashed
cap = cv2.VideoCapture(squashed_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Writer per video restaurato
out_restored = cv2.VideoWriter(restored_video_path, fourcc, fps, (original_width, original_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converti il frame da BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Converti frame in tensor PyTorch
    frame_tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Ripristina la risoluzione originale
    restored_tensor = F.interpolate(frame_tensor, size=(original_height, original_width), mode="bilinear")

    # Converti tensor in immagine
    restored_frame = (restored_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype("uint8")

    # Converti da RGB a BGR per OpenCV
    restored_frame_bgr = cv2.cvtColor(restored_frame, cv2.COLOR_RGB2BGR)

    # Scrivi il frame nel video restaurato
    out_restored.write(restored_frame_bgr)

cap.release()
out_restored.release()

print(f"Video ripristinato salvato in: {restored_video_path}")
