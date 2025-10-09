import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import glob
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter1d

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# per MS-SSIM: pip install pytorch_msssim
from pytorch_msssim import ms_ssim
# per CW-SSIM: pip install dtcwt
import dtcwt
import torch
import math
import time

#import cupy as cp
#from cupyx.scipy.ndimage import label as cp_label


#@riccardo
def min_max_normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

#@riccardo
def accuracy_per_threshold(thresholds, y_true, y_scores):
    accuracies = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)
    
    return np.array(accuracies)

def load_video_frames(video_path):
    """
    Carica i frame da un video e li restituisce in una lista.
    I frame vengono convertiti da BGR (formato OpenCV) a RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    if not cap.isOpened():
        print(f"Errore nell'aprire il video: {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Converti il frame in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    if frames==0:
        print(f"Errore nel caricamento dei frame dal video: {video_path}")
    else:
        print(f"Caricati {len(frames)} frame dal video: {video_path}")
    return frames

def compute_mse(frame1, frame2):
    """
    Calcola il Mean Squared Error (MSE) tra due frame (numpy arrays) dello stesso shape.
    """
    try:
        diff = frame1.astype(np.float32) - frame2.astype(np.float32)
        mse = np.mean(diff ** 2)
    except:
        print(f"Errore nel calcolo dell'MSE tra frame di dimensioni {frame1.shape} e {frame2.shape}")
        mse = 0
    return mse
def MSE_pesata_gpu(frame1: np.ndarray,
                   frame2: np.ndarray,
                   threshold: float = 5,
                   min_block_size: int = 25,
                   weight_by_size: bool = True,
                   exp: int = 1) -> float:
    if cp is None:
        raise ImportError("Per usare la versione GPU installa cupy: pip install cupy")

    # 1) sposto su GPU
    f1 = cp.asarray(frame1, dtype=cp.float32)
    f2 = cp.asarray(frame2, dtype=cp.float32)

    # 2) diff grayscale
    diff = cp.abs(f1 - f2)
    if diff.ndim == 3:
        diff_gray = diff.mean(axis=2)
    else:
        diff_gray = diff

    # 3) soglia + labeling
    binary = (diff_gray > threshold).astype(cp.int32)
    labels, num_features = cp_label(binary)

    # 4) bincount su GPU
    counts = cp.bincount(labels.ravel(), minlength=num_features+1)
    valid = counts >= min_block_size

    # 5) pesi
    if exp > 1:
        weights = counts * cp.log(counts + 1e-6)
    else:
        weights = counts.astype(cp.float32)
    if not weight_by_size:
        weights[:] = 1.0
    weights[~valid] = 0.0

    # 6) MSE
    weight_mask = weights[labels]
    total_w = weight_mask.sum()
    if total_w == 0:
        return 0.0

    diff_sq = diff_gray**2
    mse = (diff_sq * weight_mask).sum() / total_w
    return float(mse.get())  # scarica su CPU

def MSE_pesata(frame1: np.ndarray,
               frame2: np.ndarray,
               threshold: float = 5,
               min_block_size: int = 25,
               weight_by_size: bool = True,
               exp=1) -> float:
    """
    Calcola un MSE pesato che penalizza solo blocchi spaziali di differenza:
      - threshold: soglia di significatività (diff > threshold → differenza “attiva”)
      - min_block_size: dimensione minima di un blocco connesso (in pixel)
      - weight_by_size: se True, ogni blocco ha peso = area; altrimenti peso = 1

    Restituisce 0.0 se non ci sono blocchi significativi.
    """

    # 1) controlli di base
    if frame1.shape != frame2.shape:
        raise ValueError("frame1 e frame2 devono avere la stessa shape")
    
    # 2) differenza assoluta e conversione a singolo canale
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    if diff.ndim == 3:
        # media canali (RGB → grayscale)
        diff_gray = diff.mean(axis=2)
    else:
        diff_gray = diff

    # 3) mappe binaria delle differenze significative
    binary_diff = (diff_gray > threshold).astype(np.uint8)

    # 4) component labeling
    num_labels, labels = cv2.connectedComponents(binary_diff)

    # 5) costruzione della mappa dei pesi
    weight_mask = np.zeros_like(diff_gray, dtype=np.float32)
    for lab in range(1, num_labels):
        mask_lab = (labels == lab)
        area = int(mask_lab.sum())
        if area >= min_block_size:
            if exp >1:
                weight_mask[mask_lab] = area*math.log(area) if weight_by_size else 1.0
            else:
                weight_mask[mask_lab] = area if weight_by_size else 1.0

    total_weight = weight_mask.sum()
    if total_weight == 0:
        return 0.0

    # 6) calcolo MSE pesato
    diff_sq = diff_gray# ** 2
    weighted_mse = (diff_sq * weight_mask).sum() / total_weight

    return float(weighted_mse)

def preprocess_frame(frame, resolution: int, rgb: bool = True) -> torch.Tensor:
    
    frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0) / 255.0
    frame_resized = F.interpolate(
        frame_tensor,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False
    )
    # 5) Torna a NumPy HWC
    out_np = frame_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()          # np.ndarray
    return out_np



def ssim_loss(frame1: np.ndarray,
              frame2: np.ndarray,
              win_size: int = None,
              channel_axis: int = -1,
              data_range: float = None) -> float:
    """
    SSIM loss adattiva:
      - Se win_size non è specificato, prende min(7, min_dim) e lo rende dispari.
      - Se data_range non è specificato e il dtype è float, lo calcola come max−min di frame1.
      - Usa channel_axis per frame RGB.
    Restituisce 1 − SSIM (più alto per maggior dissimilarità).
    """
    h, w = frame1.shape[:2]
    min_dim = min(h, w)

    # determina win_size
    if win_size is None:
        ws = min(7, min_dim)
        if ws % 2 == 0:
            ws -= 1
        win_size = max(ws, 3)

    # determina data_range per immagini float
    if data_range is None:
        if np.issubdtype(frame1.dtype, np.floating):
            dr = frame1.max() - frame1.min()
            data_range = dr if (dr > 0) else 1.0
        else:
            # per int/uint8, omitto e lascio default di skimage
            data_range = None

    # parametri da passare a skimage.metrics.ssim
    kwargs = {'win_size': win_size, 'channel_axis': channel_axis}
    if data_range is not None:
        kwargs['data_range'] = data_range

    s = ssim(frame1, frame2, **kwargs)
    return 1.0 - s


def ms_ssim_loss(frame1: np.ndarray,
                  frame2: np.ndarray,
                  data_range: float = 1.0,
                  win_size: int = 7) -> float:
    """
    Multi-Scale SSIM loss adattiva:
      - Usa win_size dispari (default 7).
      - Calcola dinamicamente il numero di scale (lunghezza di `weights`) in base alla dimensione minima della frame.
      - Normalizza i pesi.
      - Restituisce 1 − MS-SSIM (più alto per maggiore dissimilarità).
    """
    # Converti a tensori Torch [1,C,H,W], normalizzati in [0,1]
    def to_tensor(img):
        t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
        if t.max() > 1.0:
            t = t / 255.0
        return t

    H, W = frame1.shape[:2]
    min_dim = min(H, W)

    # peso di default (5 scale)
    default_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    # calcola quante scale posso usare: 
    # condizione: min_dim > (win_size - 1) * 2**(scales-1)
    max_scales = int(math.floor(math.log2(min_dim / (win_size - 1)))) + 1
    max_scales = max(1, min(max_scales, len(default_weights)))

    # prendo i primi max_scales pesi e li normalizzo
    weights = default_weights[:max_scales]
    weights = [w / sum(weights) for w in weights]

    t1 = to_tensor(frame1)
    t2 = to_tensor(frame2)

    val = ms_ssim(
        t1, 
        t2,
        data_range=data_range,
        size_average=True,
        win_size=win_size,
        weights=weights
    )
    return 1.0 - val.item()


def psnr_loss(frame1: np.ndarray, frame2: np.ndarray, data_range: float = None) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) “loss”.
    Restituisce −PSNR, quindi più alto per immagini più dissimili.
    """
    value = psnr(frame1, frame2, data_range=data_range)
    return -value


def l1_loss(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Mean Absolute Error (L1) loss.
    """
    return np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))



def cw_ssim_loss(frame1: np.ndarray,
                 frame2: np.ndarray,
                 nlevels: int = 5) -> float:
    """
    Complex Wavelet SSIM (CW-SSIM) loss per immagini RGB.
    Calcola 1 − CW-SSIM index mediato su canali, scale e orientazioni.
    """
    # Assicurati che siano float64
    img1 = frame1.astype(np.float64)
    img2 = frame2.astype(np.float64)
    H, W, C = img1.shape

    # Inizializza la trasformata
    t = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')

    all_vals = []

    # Applica la DT-CWT canale per canale
    for c in range(C):
        c1 = img1[:, :, c]
        c2 = img2[:, :, c]
        coeffs1 = t.forward(c1, nlevels=nlevels)
        coeffs2 = t.forward(c2, nlevels=nlevels)

        eps = 1e-8
        # per ogni livello di dettaglio
        for hp1, hp2 in zip(coeffs1.highpasses, coeffs2.highpasses):
            # hp forma: (H_l, W_l, orientazioni)
            # somma spazio e calcola CW-SSIM per ogni orientazione
            num = 2 * np.abs((hp1 * np.conj(hp2)).sum(axis=(0,1))) + eps
            den = (np.abs(hp1)**2 + np.abs(hp2)**2).sum(axis=(0,1)) + eps
            all_vals.extend((num/den).tolist())

    # indice medio su canali+scale+orientazioni
    cwssim_index = float(np.mean(all_vals))
    return 1.0 - cwssim_index



def compute_loss(frame1, frame2, wmse=(0,0,1)): #wmse usato per definire la loss, puo essere una stringa o una tupla
    #print(" frame1 shape:", frame1.shape)
    #print(" frame2 shape:", frame2.shape)
    
    if isinstance(wmse,str):
        frame1 = preprocess_frame(frame1, resolution=128, rgb=True) 
        frame2 = preprocess_frame(frame2, resolution=128, rgb=True)
        if wmse == "ssim":
            return ssim_loss(frame1, frame2)
        elif wmse == "ms_ssim":
            return ms_ssim_loss(frame1, frame2)
        elif wmse == "psnr":
            return psnr_loss(frame1, frame2)
        elif wmse == "l1":
            return l1_loss(frame1, frame2)
        elif wmse == "cw_ssim":
            return cw_ssim_loss(frame1, frame2)
        else:
            raise ValueError(f"Unknown loss type: {wmse}")
    
    elif isinstance(wmse, tuple):
        if wmse[0]>0:
            #frame1 = preprocess_frame(frame1, resolution=512, rgb=True)
            #frame2 = preprocess_frame(frame2, resolution=512, rgb=True)
            return MSE_pesata(frame1, frame2, threshold=wmse[1], min_block_size=wmse[0], weight_by_size=True, exp=wmse[2])
        else:
            frame1 = preprocess_frame(frame1, resolution=128, rgb=True)
            frame2 = preprocess_frame(frame2, resolution=128, rgb=True)
            return compute_mse(frame1, frame2)
        
    else:
        print("Errore: wmse deve essere una stringa o una tupla riconosciuta")

def process_video_pair(video_number, generated_dir, original_dir, labels_dir, output_dir, graph=True,normalize=True, wmse=(0,0,0)):
    """
    Carica i video generati e originali usando il video_number, calcola l'MSE frame per frame,
    estrae le posizioni dei frame anomali da un file .npy e salva un grafico in output con
    l'area rossa evidenziata nei punti di anomalie.
    """
    # Costruisci i percorsi per i due video (es. "18.avi") e per il file delle etichette
    generated_video = generated_dir / f"{video_number}.avi"
    original_video = original_dir / f"{video_number}.avi"
    labels_file = labels_dir / f"{video_number}.npy"  # aggiungere in testa  01_ per avenue
    
    if not generated_video.exists() or not original_video.exists():
        print(f"File non trovato per il video {video_number}")
        return

    # Carica i frame dai video
    frames_gen = load_video_frames(generated_video)
    frames_orig = load_video_frames(original_video)
    
    if not frames_gen or not frames_orig:
        print(f"Errore nel caricamento dei frame per il video {video_number}")
        return

    n_frames = min(len(frames_gen), len(frames_orig))
    #frames_gen = frames_gen[:n_frames]
    frames_orig = frames_orig[:n_frames]

    # Carica i dati delle anomalie, se disponibili
    if not labels_file.exists():
        print(f"File delle anomalie non trovato: {labels_file}")
        anomalous_indices = []
    else:
        data = np.load(labels_file)
        print(f"Numero totale di frame originali: {data.shape[0]}")
        print(f"Numero totale di frame generati: {len(frames_gen)}")
        total_sum = np.sum(data)
        print(f"Il numero di frame anomali è: {total_sum}")
        anomalous_indices = np.where(data[:n_frames] == 1)[0]
        print(f"I frame anomali sono:\n {anomalous_indices}")

    # Calcola l'MSE per ogni frame
    #mse_values = [compute_mse(f_orig, f_gen) for f_orig, f_gen in zip(frames_orig, frames_gen)]
    
    #@riccardo nuovo MSE con stride 15
    stride= 15
    mse_values = []
    for idx in range(0, len(frames_orig)):
        f_orig = frames_orig[idx]
        f_gen = frames_gen[idx]
        
        mse1 = compute_loss(f_orig, f_gen, wmse=wmse)
        mse2 = 0
        
        if idx>15:
            gen_index=(2*stride*(idx//stride))+idx%stride
            try:
                mse1= compute_loss(f_orig, frames_gen[gen_index-stride],wmse=wmse)
            except:
                print(f"errore in mse1, cerco frame {gen_index-stride} su {len(frames_gen)}, setto mse1=0")
                mse1=0
            if (gen_index < len(frames_gen)):
                mse2= compute_loss(f_orig, frames_gen[gen_index],wmse=wmse)

        mse_values.append(max(mse1, mse2))
        #print(f"Frame {idx}: MSE = {mse:.2f}")

    if(graph):
        # Crea il grafico dell'MSE
        plt.figure(figsize=(10,6))
        plt.plot(mse_values, marker='o', linestyle='-', label="MSE")
        plt.xlabel("Frame")
        plt.ylabel("MSE")
        plt.title(f"MSE frame per frame - Video {video_number}")
        plt.grid(True)

        # Evidenzia le anomalie: per ogni frame anomalo, colora di rosso l'area corrispondente
        for idx in anomalous_indices:
            plt.axvspan(idx - 0.5, idx + 0.5, color='red', alpha=0.3)
        
        plt.legend()

        # Salva il grafico come immagine JPEG
        output_file = output_dir / f"mse_plot_{video_number}.jpeg"
        plt.savefig(output_file)
        plt.close()
        print(f"Grafico salvato in: {output_file}")
    
    #@riccardo aggiunto ritorno MSE normalizzato ed array ground truth
    
    if normalize:
        MSE_normalized = min_max_normalize(mse_values)
    else:
        MSE_normalized = mse_values
    
    ground_truth = np.zeros(len(mse_values))
    
    for idx in anomalous_indices:
        ground_truth[idx] = 1
    
    return MSE_normalized, ground_truth

def plot_normalized(mse_values,
                    ground_truth,
                    threshold,
                    output_path,
                    annotations=None,
                    title="MSE per Frame - Video 01",
                    figsize=(40, 6),
                    normalize=True):
    """
    Disegna la MSE frame-by-frame come linea, aree rosse per ground_truth == 1,
    linea di threshold, annotazioni verticali, e salva in JPEG.

    Args:
        mse_values (list or np.ndarray): valori MSE per ciascun frame.
        ground_truth (list or np.ndarray): lista binaria (0/1) per ciascun frame.
        threshold (float): valore di soglia per linea orizzontale.
        output_path (str): percorso completo (nome file .jpg) dove salvare il grafico.
        annotations (list of dict, optional): es. [{'Start': 100}, {'End': 500}], 
            per ogni dict verrà tracciata una linea verticale a frame e nominata.
        title (str): titolo del grafico.
        figsize (tuple): dimensione della figura (width, height) in pollici.
    """
    mse = np.array(mse_values)
    gt  = np.array(ground_truth)
    frames = np.arange(len(mse))

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Line plot senza marker
    ax.plot(frames, mse, linestyle='-', linewidth=1, label='MSE')

    # 2) Linea di threshold
    ax.axhline(y=threshold, color='green', linestyle='--',
               linewidth=2, label=f'Threshold = {threshold:.2f}')

    # 3) Aree rosse per blocchi di ground_truth == 1
    in_region = False
    for idx, val in enumerate(gt):
        if val == 1 and not in_region:
            start = idx
            in_region = True
        elif val == 0 and in_region:
            ax.axvspan(start, idx - 1, color='red', alpha=0.2, linewidth=0)
            in_region = False
    if in_region:
        ax.axvspan(start, len(gt) - 1, color='red', alpha=0.2, linewidth=0)

    # 4) Annotazioni verticali
    if annotations:
        ylim = ax.get_ylim()
        y_text = ylim[1] * 0.95
        for ann in annotations:
            #print("Ann :"   , ann)
            name =ann[0]
            frame = int(ann[1])
            ax.axvline(x=frame, color='black', linestyle='--', linewidth=1)
            ax.text(frame + 1, y_text, name,
                    rotation=90,
                    verticalalignment='top',
                    fontsize=9,
                    backgroundcolor='white',
                    alpha=0.7)

    # Etichette, titolo e griglia
    ax.set_xlabel("Frame")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Salvataggio in JPEG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    
def find_best_f1_threshold(y_true, y_probs):
    """
    Trova la soglia che massimizza l'F1-score.
    Args:
        y_true (array-like): Etichette vere (0 o 1).
        y_probs (array-like): Probabilità predette per la classe positiva.
    Returns:
        tuple: Soglia ottimale e F1-score corrispondente.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_threshold, best_f1

def process_all(generated_dir,original_dir, labels_dir, output_dir,multiple_graphs=False,roc_graph=False, normalize=True, wmse=(0,0,0), smoothing=0):
    
    generated_dir = Path(generated_dir)
    original_dir = Path(original_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    #@riccardo creo lista complessiva MSE e ground truth
    mse_tot = []
    gt_tot = []
    video_starting_frame=[]
    
    mse_ind=[]
    gt_ind=[]
    
    mse_no_first2=[]
    gt_no_first2=[]
    
    data_unit=""
    
    # Itera su tutti i file .avi nella directory dei video generati
    for video_file in tqdm(list(generated_dir.glob("*.avi"))):
        video_number = video_file.stem  # Estrai il numero (es. "18" da "18.avi")
        mse_par, gt_par = process_video_pair(video_number, generated_dir, original_dir, labels_dir, output_dir,graph=multiple_graphs, normalize=normalize, wmse=wmse)
        #print(f"Video {video_number} - MSE: {mse_par}, GT: {gt_par}")
        mse_tot.extend(mse_par)
        gt_tot.extend(gt_par)
        
        mse_ind.append(mse_par)
        gt_ind.append(gt_par)
        
        mse_no_first2.extend(mse_par[2:])
        gt_no_first2.extend(gt_par[2:]) 
        
        if video_starting_frame == []:
            video_starting_frame.append((video_number,len(mse_par)))
        else:
            video_starting_frame.append((video_number,len(mse_par)+video_starting_frame[-1][1]))
        
    if smoothing>0:
        mse_tot = gaussian_filter1d(mse_tot, sigma=smoothing)
        mse_no_first2 = gaussian_filter1d(mse_no_first2, sigma=smoothing)
        
        for i in range(len(mse_ind)):
            mse_ind[i] = gaussian_filter1d(mse_ind[i], sigma=smoothing)
    
    all_test_mse= [mse_tot, mse_no_first2]
    all_test_gt= [gt_tot, gt_no_first2]
    
    for i in range(len(all_test_mse)):
        #print(f"MSE: {all_test_mse[i]}\n GT: {all_test_gt[i]}")
        fpr, tpr, thresholds = roc_curve(all_test_gt[i], all_test_mse[i])
        accuracies = accuracy_per_threshold(thresholds, all_test_gt[i], all_test_mse[i])
        baseline_acc = accuracy_score(all_test_gt[i], np.zeros(len(all_test_mse[i]))) 
        baseline_acc_all = baseline_acc                       
        
        best_accuracy_index = np.argmax(accuracies)
        best_threshold = thresholds[best_accuracy_index]
        best_accuracy = accuracies[best_accuracy_index]
        
       
        
        best_f1_threshold, best_f1 = find_best_f1_threshold(all_test_gt[i],all_test_mse[i])
        
        data_unit= data_unit + f"Acc; {best_accuracy:.2f}; at threshold; {best_threshold:.2f} ; baseline to beat; {baseline_acc:.2f} ; best F1; {best_f1:.2f} at Treshold; {best_f1_threshold:.2f};"
        inference_type = ""
        if normalize:
            inference_type = inference_type+"_normalized"
        if isinstance(wmse,tuple) and wmse[0]>0:
            inference_type = inference_type+f"_wmse{wmse[0]}_th{wmse[1]}_new_exp{wmse[2]}"
        elif isinstance(wmse,str):
            inference_type = inference_type+f"_{wmse}"
            
        title = inference_type
            
        title_str = (
        f"{title} \n"
        f"Acc: {best_accuracy:.2f} at threshold: {best_threshold:.2f} ; "
        f"baseline to beat: {baseline_acc:.2f}\n"
        f"F1: {best_f1:.2f} at Treshold: {best_f1_threshold:.2f}"
        )

        # la prima iterazione e' quella con tutti i valori reali, nella seconda skppo i primi 2 frame di anchor
        if i>0:
            title="NoFirst2_" + title
        plot_normalized(all_test_mse[i], all_test_gt[i], best_threshold, output_dir / f"{title}.jpeg", title=title_str, annotations=video_starting_frame, normalize=normalize)
    
    #@riccardo riffacio gli step precedenti per calcolare gli score con tresholds individuali
    
    avg=[]
    
    for i in range(len(mse_ind)):
        
            fpr, tpr, thresholds = roc_curve(gt_ind[i], mse_ind[i])
            accuracies = accuracy_per_threshold(thresholds, gt_ind[i], mse_ind[i])
            baseline_acc = accuracy_score(gt_ind[i], np.zeros(len(mse_ind[i]))) 
                                    
            
            best_accuracy_index = np.argmax(accuracies)
            best_threshold = thresholds[best_accuracy_index]
            best_accuracy = accuracies[best_accuracy_index]
            avg.append(best_accuracy)
            
            best_f1_threshold, best_f1 = find_best_f1_threshold(gt_ind[i], mse_ind[i])
            
            
            inference_type = ""
            if normalize:
                inference_type = inference_type+"_normalized"
            if isinstance(wmse,tuple) and wmse[0]>0:
                inference_type = inference_type+f"_wmse{wmse[0]}_th{wmse[1]}_new_exp{wmse[2]}"
            elif isinstance(wmse,str):
                inference_type = inference_type+f"_{wmse}"
            
            title = inference_type
                
            title_str = (
            f"Graph {i} {title} \n"
            f"Acc: {best_accuracy:.2f} at threshold: {best_threshold:.2f} ; "
            f"baseline to beat: {baseline_acc:.2f}\n"
            f"F1: {best_f1:.2f} at Treshold: {best_f1_threshold:.2f}"
            )

            plot_normalized(mse_ind[i], gt_ind[i], best_threshold, output_dir / f"{i+1}_{title}.jpeg", title=title_str, normalize=normalize , figsize=(10, 6))
    
    f=open(output_dir / f"avg_accuracy_{round(np.mean(avg),2)}.txt", "w")
    f.write(f"avg accuracy: {round(np.mean(avg),2)}\nbaseline to beat: all neg {round(baseline_acc_all,2)} / all pos{1-round(baseline_acc_all,2)}\n")
    f.close()
    data_unit= data_unit + f"avg accuracy; {round(np.mean(avg),2)};baseline to beat; all neg ;{round(baseline_acc,2)} ; all pos;{1-round(baseline_acc,2)};"
    
    if roc_graph:
        # Grafico
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, tpr, label='TPR (True Positive Rate)')
        plt.plot(thresholds, fpr, label='FPR (False Positive Rate)')
        plt.plot(thresholds, accuracies, label='Accuracy values')
        plt.xlabel('Threshold')
        plt.ylabel('Values')
        plt.title('Threshold analisys\n Acc: {:.2f} at threshold: {:.2f} ; baseline to beat: {:.2f}'.format(best_accuracy, best_threshold,baseline_acc))
        plt.legend()
        plt.grid(True)
        #plt.gca().invert_xaxis()  # spesso si inverte l'asse per visualizzare meglio
        plt.savefig(output_dir / "roc_curve.jpeg")
        plt.close()    
        
    return data_unit

if __name__ == "__main__":
    # Directory dei video generati e originali
    
    #for num_video in os.listdir("/home/riccardo/RaMViD/outputs/avenue_avi"):
    #print(f"Processing video: {num_video}")
    
    #crea tutte le combinazioni di parametri 
    #auto=True
    
    sigma = (1,4,6)
    wmse = (10,5,1) # block-size, threshold
    normalize=False
    
    # @riccardo ATTENZIONE LA VARIABILE WMSE E' STATA RICICLATA PER SCEGLIERE ANCHE IL TIPO DI LOSS, QUINDI PUO ESSERE UNA TUPLA O UNA STRINGA
    
    #settaggio combinazioni a mano
    
    #       (sigma,wmse,normalize) settare wmse a (0,0,0) per usare MSE standard
    
    test = (
        
            (4,(0,0,0),False),(6,(0,0,0),False),
    
            (4,"ssim",False),(6,"ssim",False),
            (4,"ms_ssim",False),(6,"ms_ssim",False),
            (4,"psnr",False),(6,"psnr",False),
            (4,"l1",False),(6,"l1",False),
            (4,"cw_ssim",False),(6,"cw_ssim",False), 
            
            (4,(20,10,1),False),(6,(20,10,1),False),
            (2,(20,10,2),False),(4,(20,10,2),False),(6,(20,10,2),False), 
             
            
            #(4,   (0,   0,   0), True), (6,   (0,   0,   0), True),
#
            #(4,   "ssim",    True), (6,   "ssim",    True),
            #(4,   "ms_ssim", True), (6,   "ms_ssim", True),
            #(4,   "psnr",    True), (6,   "psnr",    True),
            #(4,   "l1",      True), (6,   "l1",      True),
            #(4,   "cw_ssim", True), (6,   "cw_ssim", True),
#
            #(4,   (20, 10, 1), True), (6,   (20, 10, 1), True),
            #(2,   (20, 10, 2), True), (4,   (20, 10, 2), True), (6,   (20, 10, 2), True),  


              
            
        
            )
    
    all_data = []
    step=len(test)
    i=0
    
    
    
    for e in test:
        i=i+1
        print(f"Processing test --sigma:{e[0]} --wmse:{e[1]} --norm:{e[2]}  test numero : {i}/{step}")
        sigma = e[0]
        wmse = e[1]
        normalize = e[2]
        
        multiple_graphs=True
        roc_graph=True
        
        inference_type = f"STC_536t_300i_sigma{sigma}"#"_1000/"
        if normalize:
            inference_type = inference_type+"_normalized"
        if isinstance(wmse,tuple) and wmse[0]>0:
            inference_type = inference_type+f"_wmse{wmse[0]}_th{wmse[1]}_new_exp{wmse[2]}"
        elif isinstance(wmse,str):
            inference_type = inference_type+f"_{wmse}"
        
        generated_dir = "/media/pinas/riccardo/STC_outputs_536train_300step/avi/"#"/home/riccardo/RaMViD/outputs/avenue_avi/" #"/media/pinas/riccardo/outputs_448train_300step/avenue_avi/"#"/home/riccardo/RaMViD/outputs"+inference_type+"avenue_avi/"
        original_dir =  "/home/riccardo/STC/shanghaitech/testing/frames/" # "/home/riccardo/Avenue_Dataset/testing_videos/"
        labels_dir = "/home/riccardo/STC/shanghaitech/testing/test_frame_mask/" # "/home/riccardo/Avenue_Dataset/masks/"
        
        # Puoi decidere se salvare i grafici in una cartella separata o insieme ai video generati:
        output_dir = "/home/riccardo/results_"+inference_type+"/" # oppure un'altra directory
        os.makedirs(output_dir, exist_ok=True)
        
        
        data = process_all(generated_dir, original_dir, labels_dir, output_dir,
                    multiple_graphs=False, 
                    roc_graph=False, 
                    normalize=normalize, 
                    wmse=wmse, 
                    smoothing=sigma)
        f=open(f"/home/riccardo/results_avenue_test_suite_.csv", "a")
        f.write(data+f"  {inference_type} \n")
        f.close()
    
 
  







#########################################################

# def main():
#     # Specifica i percorsi dei video
#     original_video = "/media/splash/simone/VAD_Mamba/Avenue_Dataset/testing_videos/18.avi"
#     generated_video = "/home/simone/RaMViD/outputs/multiple_frames/18.avi"

#     # Carica i frame dei due video
#     frames_orig = load_video_frames(original_video)
#     frames_gen = load_video_frames(generated_video)

#     if not frames_orig or not frames_gen:
#         print("Errore nel caricamento di uno dei video.")
#         return

#     # Controlla il numero di frame e, se necessario, taglia il video con più frame
#     n_orig = len(frames_orig)
#     n_gen = len(frames_gen)
#     print(f"Numero di frame: originale = {n_orig}, generato = {n_gen}")

#     if n_orig != n_gen:
#         min_frames = min(n_orig, n_gen)
#         print(f"Numero di frame differenti: si utilizza il minimo comune: {min_frames} frame")
#         frames_orig = frames_orig[:min_frames]
#         frames_gen = frames_gen[:min_frames]

#     # Calcola l'MSE frame per frame
#     mse_values = []
#     for idx, (f_orig, f_gen) in enumerate(zip(frames_orig, frames_gen)):
#         mse = compute_mse(f_orig, f_gen)
#         mse_values.append(mse)
#         # (Opzionale) Puoi stampare il valore per ogni frame:
#         # print(f"Frame {idx}: MSE = {mse:.2f}")

#     # Crea un grafico dell'MSE per frame
#     plt.figure(figsize=(10,6))
#     plt.plot(mse_values, marker='o', linestyle='-')
#     plt.xlabel("Frame")
#     plt.ylabel("MSE")
#     plt.title("MSE frame per frame tra video originale e generato")
#     plt.grid(True)
    
#     # Salva il grafico in JPEG (modifica il path se necessario)
#     output_graph = "/home/simone/RaMViD/outputs/multiple_frames/mse_plot_18.jpeg"
#     plt.savefig(output_graph)
#     plt.close()
#     print(f"Grafico salvato in: {output_graph}")

# if __name__ == '__main__':
#     main()
