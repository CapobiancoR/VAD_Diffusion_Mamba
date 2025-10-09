import torch

checkpoint_paths = [
    # "/media/nvme_4tb/simone_data/log/ema_0.9999_002000.pt",
    # "/media/nvme_4tb/simone_data/log/ema_0.9999_022000.pt",
    # "/media/nvme_4tb/simone_data/log/model002000.pt",
    # "/media/nvme_4tb/simone_data/log/model022000.pt",
    "/media/nvme_4tb/simone_data/log/opt002000.pt",
    "/media/nvme_4tb/simone_data/log/opt008000.pt", # questo è il migliore
    "/media/nvme_4tb/simone_data/log/opt010000.pt"
]

for path in checkpoint_paths:
    try:
        checkpoint = torch.load(path, map_location="cpu")
        print(f"Checkpoint {path} caricato correttamente.")
        if isinstance(checkpoint, dict):
            #print(f"Chiavi presenti nel checkpoint: TLDR")
            continue
        else:
            print("Checkpoint non è un dizionario. Controlla il contenuto manualmente.")
    except Exception as e:
        print(f"Errore nel caricamento del checkpoint {path}: {e}")
