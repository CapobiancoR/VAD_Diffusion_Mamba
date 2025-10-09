import os
import cv2

def frames_to_avi(root_folder: str, fps: int = 15, codec: str = 'XVID'):
    """
    Scorre tutte le sottocartelle di `root_folder` e per ciascuna
    genera un video .avi con nome uguale alla cartella.
    
    Args:
        root_folder (str): percorso della cartella contenente le sottocartelle di frame.
        fps (int): frame per secondo del video di output.
        codec (str): quattro caratteri del codec (es. 'XVID', 'MJPG', ecc.).
    """
    # Codice per il VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*codec)
    
    # Scorri tutte le sottocartelle
    for sub in sorted(os.listdir(root_folder)):
        sub_path = os.path.join(root_folder, sub)
        if not os.path.isdir(sub_path):
            continue
        
        # Prendi tutti i file .jpg ordinati per nome
        frames = [f for f in os.listdir(sub_path) if f.lower().endswith('.jpg')]
        if not frames:
            print(f"[!] Nessun frame in {sub}, skip.")
            continue
        frames.sort(key=lambda x: int(os.path.splitext(x)[0]))  # assuming names “0.jpg, 1.jpg, …”
        
        # Leggi il primo frame per avere dimensioni
        first_frame = cv2.imread(os.path.join(sub_path, frames[0]))
        if first_frame is None:
            print(f"[!] Impossibile leggere il frame {frames[0]} in {sub}, skip.")
            continue
        height, width = first_frame.shape[:2]
        
        # Crea il VideoWriter
        out_path = os.path.join(root_folder, f"{sub}.avi")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        # Scrivi tutti i frame
        for fname in frames:
            img_path = os.path.join(sub_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[!] Impossibile leggere {img_path}, salto.")
                continue
            # Se il frame non è già in BGR a 3 canali, fai la conversione:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            writer.write(img)
        
        writer.release()
        print(f"[✓] Video creato: {out_path}")

# Esempio d’uso:
if __name__ == "__main__":
    root = "/home/riccardo/STC/shanghaitech/testing/frames/"  # o la tua cartella di root
    frames_to_avi(root_folder=root, fps=15, codec='XVID')
