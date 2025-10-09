from random import sample
from PIL import Image, ImageSequence
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn.functional as F
import av
from torchvision.transforms.functional import to_tensor #@simone
import os # @simone

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, rgb=True, seq_len=25, starting_frame=None
):
    """
    For a dataset, create a generator over (videos, kwargs) pairs.

    Each video is an NCLHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which frames are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """

    # @simone We did a small change. We added an if-else condition to check whether data_dir was a directory, or a single file.
    # When a directory, we use the standard implementation proposed by RaMViD. 
    # When a file, we implement a mechanism to choose the specific starting frame. This allows for a greater control for our experiments.
    if os.path.isdir(data_dir): # CASE 1
        if not data_dir:
            raise ValueError("unspecified data directory or file. Please enter a valid path.")
        all_files = _list_video_files_recursively(data_dir) # returns a list of all the files contained in the directory
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        
        entry = all_files[0].split(".")[-1] # check the extension of the file
        if entry in ["avi", "mp4"]:
            dataset = VideoDataset_mp4(
                image_size,
                all_files,
                classes=classes,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                rgb=rgb,
                seq_len=seq_len
            )
        elif entry in ["gif"]:
            dataset = VideoDataset_gif(
                image_size,
                all_files,
                classes=classes,
                shard=MPI.COMM_WORLD.Get_rank(),
                num_shards=MPI.COMM_WORLD.Get_size(),
                rgb=rgb,
                seq_len=seq_len
            )
        if deterministic:
            # Load using a fixed order
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True
            )
        else:
            # Load by shuffling the data
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True
            )
        while True:
            yield from loader

    elif os.path.isfile(data_dir) and data_dir.split(".")[-1].lower() in ["avi", "mp4"]:
        if isinstance(starting_frame, list) and len(starting_frame) > 1:
            all_files = [data_dir] * len(starting_frame) # replicate the single path for each element
        else:
            all_files = [data_dir] # fallback if single integer or None

        dataset = VideoDataset_mp4(
            image_size,
            all_files,    # @simone each item in the dataset is the SAME file path but with a different start!
            classes=None,
            shard=0,
            num_shards=1,
            rgb=rgb,
            seq_len=seq_len,
            starting_frame=starting_frame  # @simone this will be like: 0,25,50,100 (that will be transformed in a list of starting frames)
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,
            drop_last=False
        )

        for batch in loader:
            video_data = batch[0]
            yield video_data, None


def _list_video_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["gif", "avi", "mp4"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_video_files_recursively(full_path))
    return results

# class VideoDataset_mp4(Dataset):
#     ## @simone classe originale usata da RaMVID
#     def __init__(self, resolution, video_paths, classes=None, shard=0, num_shards=1, rgb=True, seq_len=20):
#         super().__init__()
#         self.resolution = resolution
#         self.local_videos = video_paths[shard:][::num_shards]
#         self.local_classes = None if classes is None else classes[shard:][::num_shards]
#         self.rgb = rgb
#         self.seq_len = seq_len

#     def __len__(self):
#         return len(self.local_videos)

#     def __getitem__(self, idx):
#         path = self.local_videos[idx]
#         arr_list = []
#         video_container = av.open(path)
#         n = video_container.streams.video[0].frames
#         frames = [i for i in range(n)]
#         if n > self.seq_len:
#             start = np.random.randint(0, n-self.seq_len)
#             frames = frames[start:start + self.seq_len]
#         for id, frame_av in enumerate(video_container.decode(video=0)):
#         # We are not on a new enough PIL to support the `reducing_gap`
#         # argument, which uses BOX downsampling at powers of two first.
#         # Thus, we do it by hand to improve downsample quality.
#             if (id not in frames):
#                 continue
#             frame = frame_av.to_image()
#             while min(*frame.size) >= 2 * self.resolution:
#                 frame = frame.resize(
#                     tuple(x // 2 for x in frame.size), resample=Image.BOX
#                 )
#             scale = self.resolution / min(*frame.size)
#             frame =frame.resize(
#                 tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
#             )

#             if self.rgb:
#                 arr = np.array(frame.convert("RGB"))
#             else:
#                 arr = np.array(frame.convert("L"))
#                 arr = np.expand_dims(arr, axis=2)
#             crop_y = (arr.shape[0] - self.resolution) // 2
#             crop_x = (arr.shape[1] - self.resolution) // 2
#             arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
#             arr = arr.astype(np.float32) / 127.5 - 1
#             arr_list.append(arr)
#         arr_seq = np.array(arr_list)
#         arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        
#         # Se mancano frame, fill in missing frames with 0s
#         if arr_seq.shape[1] < self.seq_len:
#             required_dim = self.seq_len - arr_seq.shape[1]
#             fill = np.zeros((3, required_dim, self.resolution, self.resolution))
#             arr_seq = np.concatenate((arr_seq, fill), axis=1)
#         out_dict = {}
#         if self.local_classes is not None:
#             out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
#         return arr_seq, out_dict

class VideoDataset_mp4(Dataset):
    # @simone Nuova classe con Interpolate invece che Crop.
    # NOTE: ovviamente andrà invertita per ri-ottneere il video nel formato originale. Per info aggiuntive, guarda il file /home/simone/RaMViD/test_dimensioni.py
    def __init__(self, resolution, video_paths, classes=None, shard=0, num_shards=1, rgb=True, seq_len=25, starting_frame=None):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.rgb = rgb
        self.seq_len = seq_len
        self.starting_frame = starting_frame  # self.starting_frame might be int, list, or None

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]

        arr_list = []
        video_container = av.open(path)
        n = video_container.streams.video[0].frames
        frames = [i for i in range(n)]
        
        if not self.starting_frame:
            # Se starting_frame è None o vuoto, usiamo la logica casuale.
            if n > self.seq_len:
                start = np.random.randint(0, n - self.seq_len)
                frames = frames[start : start + self.seq_len]
        else:
            # starting_frame può essere una lista o un singolo intero
            if isinstance(self.starting_frame, list):
                # Ogni indice idx ha il proprio start
                start = self.starting_frame[idx]
            else:
                # È un singolo intero
                start = self.starting_frame

            end = min(start + self.seq_len, n)
            frames = frames[start:end]
            # print(f"Start: {start}; End: {end}: Len(frames): {len(frames)}; Seq_len: {self.seq_len}") #@simone

            if len(frames) < self.seq_len:
                raise ValueError(f"Mismatch in the length of the video! Check again the actual length and the desired seq_len")   

        for id, frame_av in enumerate(video_container.decode(video=0)):
            if (id not in frames):
                continue
            frame = frame_av.to_image()  # frame PIL

            # Converte in RGB o L
            if self.rgb:
                frame_np = np.array(frame.convert("RGB"), dtype=np.float32)
            else:
                frame_np = np.array(frame.convert("L"), dtype=np.float32)
                frame_np = np.expand_dims(frame_np, axis=2)

            # Ora frame_np è in [0,255], convertiamo in [0,1] prima di interpolare
            frame_tensor = torch.from_numpy(frame_np).permute(2,0,1).unsqueeze(0) / 255.0

            # Interpolazione a dimensioni (resolution, resolution)
            frame_resized = F.interpolate(frame_tensor, size=(self.resolution, self.resolution), mode="bilinear")

            # Torna a NumPy: [H, W, C], ancora in [0,1]
            arr = frame_resized.squeeze(0).permute(1, 2, 0).numpy()

            # Riportiamo in [0,255] per usare la stessa normalizzazione del codice originale
            arr = arr * 255.0

            # Normalizzazione come prima: da [0,255] a [-1,1]
            arr = arr.astype(np.float32) / 127.5 - 1.0

            arr_list.append(arr)

        arr_seq = np.array(arr_list)
        # arr_seq ha forma (seq_len, H, W, C) -> (C, seq_len, H, W)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])

        # Se mancano frame, riempi con zeri
        if arr_seq.shape[1] < self.seq_len:
            required_dim = self.seq_len - arr_seq.shape[1]
            fill = np.zeros((3, required_dim, self.resolution, self.resolution), dtype=np.float32)
            arr_seq = np.concatenate((arr_seq, fill), axis=1)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return arr_seq, out_dict


class VideoDataset_gif(Dataset):
    def __init__(self, resolution, video_paths, classes=None, shard=0, num_shards=1, rgb=True, seq_len=20):
        super().__init__()
        self.resolution = resolution
        self.local_videos = video_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.rgb = rgb
        self.seq_len = seq_len

    def __len__(self):
        return len(self.local_videos)

    def __getitem__(self, idx):
        path = self.local_videos[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_videos = Image.open(f)
            arr_list = []
            for frame in ImageSequence.Iterator(pil_videos):

            # We are not on a new enough PIL to support the `reducing_gap`
            # argument, which uses BOX downsampling at powers of two first.
            # Thus, we do it by hand to improve downsample quality.
                while min(*frame.size) >= 2 * self.resolution:
                    frame = frame.resize(
                        tuple(x // 2 for x in frame.size), resample=Image.BOX
                    )
                scale = self.resolution / min(*frame.size)
                frame =frame.resize(
                    tuple(round(x * scale) for x in frame.size), resample=Image.BICUBIC
                )

                if self.rgb:
                    arr = np.array(frame.convert("RGB"))
                else:
                    arr = np.array(frame.convert("L"))
                    arr = np.expand_dims(arr, axis=2)
                crop_y = (arr.shape[0] - self.resolution) // 2
                crop_x = (arr.shape[1] - self.resolution) // 2
                arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
                arr = arr.astype(np.float32) / 127.5 - 1
                arr_list.append(arr)
        arr_seq = np.array(arr_list)
        arr_seq = np.transpose(arr_seq, [3, 0, 1, 2])
        if arr_seq.shape[1] > self.seq_len:
            start = np.random.randint(0, arr_seq.shape[1]-self.seq_len)
            arr_seq = arr_seq[:,start:start + self.seq_len]
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return arr_seq, out_dict
