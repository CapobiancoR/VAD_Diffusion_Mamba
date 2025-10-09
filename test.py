import os
import shutil
from typing import Union


def keep_recent_items(folder_path: Union[str, os.PathLike], n: int = 9) -> None:
    """
    Mantiene i n elementi più recenti in una cartella e rimuove tutti gli altri.

    Args:
        folder_path: percorso della cartella da pulire.
        n: numero di elementi da mantenere (default 9).

    Raises:
        FileNotFoundError: se folder_path non esiste o non è una cartella.
        PermissionError: se non ci sono permessi sufficienti per rimuovere files.
    """
    # Verifica che il percorso esista e sia una directory
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"'{folder_path}' non esiste o non è una cartella valida.")

    # Ottiene lista completa di elementi (file e directory) con il relativo percorso assoluto
    all_items = [os.path.join(folder_path, name) for name in os.listdir(folder_path)]
    
    # Ordina per tempo di modifica (i più nuovi primi)
    sorted_items = sorted(
        all_items,
        key=lambda path: os.path.getmtime(path),
        reverse=True
    )

    # Se ci sono più di n elementi, elimina quelli oltre i primi n
    to_delete = sorted_items[n:]
    print('Removing the following items:', to_delete) 
    for path in to_delete:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            # print(f"Rimosso: {path}")
            #print(f"Removed: {path}") 
            print(f"Removed: {path}")  # Log the removal
        except Exception as e:
            # Qui potresti voler loggare l'errore o rilanciarlo
            print(f"Errore rimuovendo '{path}': {e}")
            
# Esempio d'uso:
if __name__ == "__main__":
    folder = "/home/riccardo/checkpoints_STC"
    keep_recent_items(folder, n=9)
