"""
Script per generare video di attivazione pixel-by-pixel usando MSE_pesata.
Prende due video in input (originale e generato) e produce un video che mostra
l'attivazione dei pixel basata sulla blob loss pesata.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse
import math


def load_video_frames(video_path):
    """
    Carica i frame da un video e li restituisce in una lista.
    I frame vengono convertiti da BGR (formato OpenCV) a RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {video_path}")
        return frames, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Converti il frame in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    if len(frames) == 0:
        print(f"Errore nel caricamento dei frame dal video: {video_path}")
    else:
        print(f"Caricati {len(frames)} frame dal video: {video_path}")
    
    return frames, fps, frame_size


def compute_pixel_activation_map(frame1: np.ndarray,
                                  frame2: np.ndarray,
                                  threshold: float = 5,
                                  min_block_size: int = 25,
                                  weight_by_size: bool = True,
                                  exp: int = 1) -> np.ndarray:
    """
    Calcola una mappa di attivazione pixel-by-pixel basata sulla MSE_pesata.
    Restituisce una mappa 2D dove ogni pixel ha un valore di attivazione.
    
    Args:
        frame1, frame2: frame da confrontare (RGB)
        threshold: soglia di significatività
        min_block_size: dimensione minima di un blocco connesso
        weight_by_size: se True, peso = area; altrimenti peso = 1
        exp: esponente per il peso (1=lineare, >1=super-lineare)
    
    Returns:
        activation_map: mappa 2D con valori di attivazione per pixel
    """
    # 1) Controlli di base
    if frame1.shape != frame2.shape:
        raise ValueError("frame1 e frame2 devono avere la stessa shape")
    
    # 2) Differenza assoluta e conversione a singolo canale
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    if diff.ndim == 3:
        # Media canali (RGB → grayscale)
        diff_gray = diff.mean(axis=2)
    else:
        diff_gray = diff

    # 3) Mappa binaria delle differenze significative
    binary_diff = (diff_gray > threshold).astype(np.uint8)

    # 4) Component labeling
    num_labels, labels = cv2.connectedComponents(binary_diff)

    # 5) Costruzione della mappa dei pesi
    weight_mask = np.zeros_like(diff_gray, dtype=np.float32)
    for lab in range(1, num_labels):
        mask_lab = (labels == lab)
        area = int(mask_lab.sum())
        if area >= min_block_size:
            if exp > 1:
                weight_mask[mask_lab] = area * math.log(area) if weight_by_size else 1.0
            else:
                weight_mask[mask_lab] = area if weight_by_size else 1.0

    # 6) Calcola la mappa di attivazione pixel-by-pixel
    # Qui usiamo diff_gray (differenza assoluta) moltiplicata per il peso
    activation_map = diff_gray * weight_mask
    
    return activation_map


def normalize_activation_map(activation_map: np.ndarray, 
                             method: str = 'minmax') -> np.ndarray:
    """
    Normalizza la mappa di attivazione per la visualizzazione.
    
    Args:
        activation_map: mappa di attivazione raw
        method: 'minmax' o 'percentile'
    
    Returns:
        mappa normalizzata in range [0, 255] come uint8
    """
    if method == 'minmax':
        min_val = activation_map.min()
        max_val = activation_map.max()
        if max_val - min_val > 0:
            normalized = ((activation_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(activation_map, dtype=np.uint8)
    
    elif method == 'percentile':
        p_low = np.percentile(activation_map, 1)
        p_high = np.percentile(activation_map, 99)
        clipped = np.clip(activation_map, p_low, p_high)
        if p_high - p_low > 0:
            normalized = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(activation_map, dtype=np.uint8)
    
    else:
        raise ValueError(f"Metodo di normalizzazione non riconosciuto: {method}")
    
    return normalized


def apply_colormap(activation_map: np.ndarray, 
                   colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Applica una colormap alla mappa di attivazione.
    
    Args:
        activation_map: mappa normalizzata (uint8)
        colormap: colormap OpenCV (es. COLORMAP_JET, COLORMAP_HOT, COLORMAP_VIRIDIS)
    
    Returns:
        immagine RGB con colormap applicata
    """
    colored = cv2.applyColorMap(activation_map, colormap)
    # Converti da BGR a RGB
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored_rgb


def create_overlay(original_frame: np.ndarray,
                   activation_colored: np.ndarray,
                   alpha: float = 0.5) -> np.ndarray:
    """
    Crea un overlay tra il frame originale e la mappa di attivazione colorata.
    
    Args:
        original_frame: frame originale RGB
        activation_colored: mappa di attivazione con colormap RGB
        alpha: trasparenza (0=solo originale, 1=solo attivazione)
    
    Returns:
        frame con overlay
    """
    # Assicurati che entrambi siano float per il blending
    orig_float = original_frame.astype(np.float32)
    activ_float = activation_colored.astype(np.float32)
    
    # Blend
    blended = ((1 - alpha) * orig_float + alpha * activ_float).astype(np.uint8)
    
    return blended


def process_videos(original_video_path: str,
                   generated_video_path: str,
                   output_video_path: str,
                   threshold: float = 5,
                   min_block_size: int = 25,
                   weight_by_size: bool = True,
                   exp: int = 1,
                   normalization_method: str = 'minmax',
                   colormap: int = cv2.COLORMAP_JET,
                   output_mode: str = 'activation',
                   overlay_alpha: float = 0.5):
    """
    Processa due video e genera un video di attivazione.
    
    Args:
        original_video_path: percorso video originale
        generated_video_path: percorso video generato
        output_video_path: percorso video output
        threshold: soglia per MSE_pesata
        min_block_size: dimensione minima blob
        weight_by_size: se pesare per dimensione
        exp: esponente per il peso
        normalization_method: 'minmax' o 'percentile'
        colormap: colormap OpenCV da usare
        output_mode: 'activation', 'overlay', 'sidebyside'
        overlay_alpha: trasparenza per overlay (0-1)
    """
    # Carica i video
    print("Caricamento video originale...")
    frames_orig, fps_orig, size_orig = load_video_frames(original_video_path)
    
    print("Caricamento video generato...")
    frames_gen, fps_gen, size_gen = load_video_frames(generated_video_path)
    
    if not frames_orig or not frames_gen:
        print("Errore nel caricamento dei video.")
        return
    
    # Usa il minimo numero di frame
    n_frames = min(len(frames_orig), len(frames_gen))
    frames_orig = frames_orig[:n_frames]
    frames_gen = frames_gen[:n_frames]
    
    print(f"Processando {n_frames} frame...")
    
    # Determina dimensioni output
    if output_mode == 'sidebyside':
        output_width = size_orig[0] * 2
        output_height = size_orig[1]
    else:
        output_width = size_orig[0]
        output_height = size_orig[1]
    
    # Inizializza il writer video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = fps_orig if fps_orig else 25.0
    out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                         (output_width, output_height))
    
    # Processa frame per frame
    for idx in tqdm(range(n_frames), desc="Generazione video attivazione"):
        frame_orig = frames_orig[idx]
        frame_gen = frames_gen[idx]
        
        # Calcola mappa di attivazione
        activation_map = compute_pixel_activation_map(
            frame_orig, frame_gen,
            threshold=threshold,
            min_block_size=min_block_size,
            weight_by_size=weight_by_size,
            exp=exp
        )
        
        # Normalizza
        activation_normalized = normalize_activation_map(
            activation_map, 
            method=normalization_method
        )
        
        # Applica colormap
        activation_colored = apply_colormap(activation_normalized, colormap)
        
        # Genera output in base alla modalità
        if output_mode == 'activation':
            output_frame = activation_colored
        
        elif output_mode == 'overlay':
            output_frame = create_overlay(frame_orig, activation_colored, overlay_alpha)
        
        elif output_mode == 'sidebyside':
            # Affianca frame originale e attivazione
            output_frame = np.hstack([frame_orig, activation_colored])
        
        else:
            raise ValueError(f"Modalità output non riconosciuta: {output_mode}")
        
        # Converti RGB a BGR per OpenCV
        output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
        
        # Scrivi frame
        out.write(output_frame_bgr)
    
    out.release()
    print(f"Video salvato in: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Genera video di attivazione pixel-by-pixel usando MSE_pesata'
    )
    
    parser.add_argument('--original', type=str, required=True,
                       help='Percorso video originale')
    parser.add_argument('--generated', type=str, required=True,
                       help='Percorso video generato')
    parser.add_argument('--output', type=str, required=True,
                       help='Percorso video output')
    parser.add_argument('--threshold', type=float, default=5,
                       help='Soglia per differenza significativa (default: 5)')
    parser.add_argument('--min_block_size', type=int, default=25,
                       help='Dimensione minima blob in pixel (default: 25)')
    parser.add_argument('--exp', type=int, default=1,
                       help='Esponente per il peso (1=lineare, >1=super-lineare)')
    parser.add_argument('--no_weight_by_size', action='store_true',
                       help='Non pesare per dimensione area')
    parser.add_argument('--normalization', type=str, default='minmax',
                       choices=['minmax', 'percentile'],
                       help='Metodo di normalizzazione (default: minmax)')
    parser.add_argument('--colormap', type=str, default='jet',
                       choices=['jet', 'hot', 'viridis', 'plasma', 'turbo'],
                       help='Colormap da usare (default: jet)')
    parser.add_argument('--mode', type=str, default='activation',
                       choices=['activation', 'overlay', 'sidebyside'],
                       help='Modalità output: activation, overlay, sidebyside (default: activation)')
    parser.add_argument('--overlay_alpha', type=float, default=0.5,
                       help='Trasparenza overlay (0-1, default: 0.5)')
    
    args = parser.parse_args()
    
    # Mappa nomi colormap a costanti OpenCV
    colormap_dict = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'turbo': cv2.COLORMAP_TURBO
    }
    
    # Processa i video
    process_videos(
        original_video_path=args.original,
        generated_video_path=args.generated,
        output_video_path=args.output,
        threshold=args.threshold,
        min_block_size=args.min_block_size,
        weight_by_size=not args.no_weight_by_size,
        exp=args.exp,
        normalization_method=args.normalization,
        colormap=colormap_dict[args.colormap],
        output_mode=args.mode,
        overlay_alpha=args.overlay_alpha
    )


if __name__ == "__main__":
    # Esempio di utilizzo diretto senza argomenti
    # Decommenta e modifica i percorsi per usare direttamente
    """
    process_videos(
        original_video_path="path/to/original.avi",
        generated_video_path="path/to/generated.avi",
        output_video_path="path/to/activation_output.avi",
        threshold=10,
        min_block_size=20,
        weight_by_size=True,
        exp=2,
        normalization_method='minmax',
        colormap=cv2.COLORMAP_JET,
        output_mode='overlay',
        overlay_alpha=0.6
    )
    """
    
    #main()
