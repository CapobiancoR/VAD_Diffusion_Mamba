#!/usr/bin/env python3
import os
import shutil

# === CONFIGURA QUI I PERCORSI ===
# Path del file .txt contenente un filename (senza estensione) per riga
list_file =  "/media/pinas/riccardo/UBnormal_video/scripts/abnormal_test_video_names.txt" #"/media/pinas/riccardo/UBnormal_video/scripts/normal_training_video_names.txt"
# Directory in cui cercare (ricorsivamente) i file
search_dir = "/media/pinas/riccardo/UBnormal_video/"
# ==================================

# Preparo la cartella 'training' accanto al TXT
base_dir = os.path.dirname(os.path.abspath(list_file))
training_dir = os.path.join(search_dir, "UB_test")
os.makedirs(training_dir, exist_ok=True)

# Leggo i nomi da cercare (senza estensione)
with open(list_file, "r", encoding="utf-8") as f:
    targets = {line.strip() for line in f if line.strip()}

print(targets)
# Scansione ricorsiva e copia
copied = 0
for root, _, files in os.walk(search_dir):
    for fname in files:
        name_no_ext, ext = os.path.splitext(fname)
        if name_no_ext in targets:
            src = os.path.join(root, fname)
            dst = os.path.join(training_dir, fname)
            try:
                shutil.copy2(src, dst)
                copied += 1
                print(f"[+] Copiato: {src} → {dst}")
            except shutil.SameFileError:
                # In rari casi di link o permessi,
                # evitiamo che lo script si fermi
                continue

print(f"\nTotale file copiati: {copied}")
