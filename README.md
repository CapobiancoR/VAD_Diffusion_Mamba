# Video Anomaly Detection with Mamba Video Diffusion
*Riccardo Capobianco*  
*Deep Learning and Applied AI – Sapienza University of Rome (2025)*

---

## 1. Introduction

Video Anomaly Detection (VAD) requires identifying spatio-temporal patterns that diverge from learned “normality.”  
This involves modeling long-range temporal dependencies, preserving spatial coherence, and capturing fine-grained motion.  
A common limitation is the reliance on short temporal contexts, which reduces robustness in complex dynamic scenes.

This work addresses VAD using an **unsupervised diffusion-based generative framework** that replaces attention with **state-space modeling (SSMs)**.  
The goal is to extend the effective temporal context without increasing computational cost.

We adapt **Random-Mask Video Diffusion (RaMViD)** to the anomaly-detection setting, replacing its attention modules with **Mamba** state-space blocks.  
Given an input video, the model reconstructs masked frames from partial observations, and **reconstruction residuals** act as anomaly indicators.  
We further improve stability through a **custom anomaly-scoring function** tailored for heterogeneous scenes.

---

## 2. Related Work

VAD methods are generally categorized into:
- **Prediction-based models**: forecast future frames or optical flow.
- **Reconstruction-based generative models**: reconstruct normal inputs using autoencoders or diffusion models.
- **Memory-based approaches**: store prototypes of “normal” events.
- **Hybrid methods**: combine prediction and memory.

Examples include:
- **Patch-level diffusion (Zhou et al.)**: predicts next-frame crops using U-Net denoisers.  
- **Attention-based reconstruction (Wang et al.)**: operates on object-level ROIs with appearance and motion branches.  
- **LGN-Net (Zhao et al.)**: combines convolution, LSTM, and memory units for spatio-temporal reasoning.

Our model improves upon these by **reconstructing up to 22 frames per generation**, significantly extending temporal context.

---

## 3. Methodology

### 3.1 Overview
We treat VAD as **conditional video infilling**:  
Given anchor frames `C` within a window of length `L`, the model reconstructs the masked frames `U`.  
We adapt RaMViD by introducing **selective masking** instead of random masking to speed up training and inference.

![Figure 1. Example input of the network with C=(0,8)](figure1.png)

---

### 3.2 Replacing Attention with State-Space Sequence Modeling
Self-attention blocks are replaced by **Bidirectional Mamba** layers (imported from *VideoMamba*).  
These layers process flattened spatio-temporal sequences `[B, L, C]` where `L = F × H × W`.  
Mamba provides **O(L)** complexity (vs. O(L²) for attention), allowing longer sequences while preserving 3D convolutional spatial priors.

---

### 3.3 Inference: Overlapping Infilling and Max Aggregation
During testing, each video is processed in windows of 30 frames with stride 15, so each frame is reconstructed twice.  
We compute the final frame anomaly score as:

\[
s_t = \max\big(L(x_t, \hat{x}_t^{(1)}),\, L(x_t, \hat{x}_t^{(2)})\big)
\]

This approach emphasizes informative reconstruction errors while discarding trivial ones on anchor frames.

---

### 3.4 Anomaly Scoring
The resulting frame-wise scores form an anomaly curve `{s_t}`.  
We evaluate using ROC-based metrics and compare various scoring methods, including **MSE**, **L1**, and **Structural Similarity (SSIM)**, against our custom **Weighted MSE (WMSE)**.

---

### 3.5 Training Protocol
Training follows a **one-class** setup using only normal videos.  
Windows have `L = 30` with fixed selective masking.  
Optimization uses the RaMViD diffusion objective, computing the denoising loss only on masked frames.

---

## 4. Experiments and Results

### 4.1 Experimental Setup
Inference uses 200 diffusion steps per clip.  
Anomaly scores are smoothed with Gaussian filtering and optionally normalized.  
Evaluation includes frame-level accuracy and average per-video accuracy.

### 4.2 Overview of Results
The model is evaluated on **Avenue**, **ShanghaiTech Campus (STC)**, and **UBnormal** datasets.  
Differences in accuracy across datasets are explained by the number of scenes and training steps per scene.

| Method | Type | Avenue | STC | UBnormal |
|:--------|:------|:------:|:----:|:---------:|
| MA-PDM (Zhou et al.) | Diffusion (patch, motion+appearance) | – | – | – |
| Making Reconstruction Great Again | Reconstruction with attention | – | – | – |
| LGN-Net (Zhao et al.) | Prediction + Memory | – | – | – |
| **Ours** | Diffusion with Mamba | – | – | – |

---

## Appendix

### 6. Notation
- **B** – Batch size  
- **C** – Number of feature channels  
- **F** – Number of frames  
- **H, W** – Frame height and width  
- **L = F × H × W** – Flattened sequence length  
- **O(L)** vs. **O(L²)** – Mamba vs. Attention complexity  

---

### 7. From Random to Selective Masking
RaMViD used random masking for generalization.  
We replace it with **deterministic selective masking**, fixing the anchor frames (first two, every 10th–11th, and last two).  
This focuses learning on the same masking pattern used at inference, improving sample quality and allowing **5× faster inference** (200 vs. 1000 diffusion steps).

---

### 8. Frame Resize
All frames are bilinearly resized to **128×128** before normalization and batching to standardize input size and reduce computational cost.

---

### 9. Why Bidirectional Mamba Blocks
Bidirectional Mamba captures both **past and future context**, improving reconstruction fidelity and anomaly discrimination.  
It maintains linear complexity and stable training while preserving local spatial inductive biases via 3D convolutions.

---

### 10. Weighted Mean Squared Error (WMSE)
Our custom WMSE emphasizes **spatially coherent anomalies** (“blobs”) rather than isolated pixel noise.

Steps:
1. Compute grayscale absolute differences.  
2. Threshold to detect active pixels.  
3. Group into connected components (blobs).  
4. Weight each blob by area or log(area).  
5. Compute the weighted average of per-pixel errors.  

This produces robust anomaly scores less sensitive to small artifacts.

---

## 5. Conclusions
This study proposes a **state-space diffusion model** for video anomaly detection, replacing attention with **Mamba** to improve temporal context and efficiency.  
Performance variations across datasets are mainly attributed to training steps per scene rather than model limitations.  
Future work includes further experiments on larger datasets and analysis of cross-scene generalization.

 _________________________________________
 # RaMViD (Original Readme)

Official code for the paper:

> [Diffusion Models for Video Prediction and Infilling](https://arxiv.org/abs/2206.07696) <br/>
> Tobias Höppe, Arash Mehrjou, Stefan Bauer, Didrik Nielsen, Andrea Dittadi <br/>
> TMLR, 2022

**Project website: https://sites.google.com/view/video-diffusion-prediction**

This code and README are based on https://github.com/openai/improved-diffusion

## Installation

Import and create the enroot container 
```
$ enroot import docker://nvcr.io#nvidia/pytorch:21.04-py3
$ enroot create --name container_name nvidia+pytorch+21.04-py3.sqsh
```
and run in addition
```
pip install torch
pip install tqdm
pip install blobfile>=0.11.0
pip install mpi4py
pip install matplotlib
pip install av 
```

## Preparing Data

Our dataloader can handle videos in the .gif, .mp4 or .av format. Create a folder with your data and simply pass `--data_dir path/to/videos` to the training script

## Training

Similar to the original code baes, will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. 

Kinetics-600:  
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 8 --microbatch 2 --seq_len 16 --max_num_mask_frames 4 --uncondition_rate 0.25";
```

BAIR:  
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 2 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 4 --microbatch 2 --seq_len 20 --max_num_mask_frames 4 --uncondition_rate 0.25";
```

UCF-101:  
```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0";
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear";
TRAIN_FLAGS="--lr 2e-5 --batch_size 8 --microbatch 2 --seq_len 16 --max_num_mask_frames 4 --uncondition_rate 0.75";
```

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python scripts/video_train.py --data_dir path/to/videos $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

You may also want to train in a distributed manner. In this case, run the same command with `mpirun`:

```
mpirun -n $NUM_GPUS python scripts/video_train.py --data_dir path/to/videos $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

if you want to continue training a model, you should add the model path to the command via `--resume_checkpoint`. In the same folder you should have the saved optimizer and ema model.

Pre-trained models:<br/>
[Model trained on Kinetics-600 (K=4, p=0.25)](https://1drv.ms/f/s!Amih97wvmSyWgosH_uJoN-BsH_RWkw?e=P0Wg8n)  <br/>
[Model trained on BAIR (K=4, p=0.25)](https://1drv.ms/f/s!Amih97wvmSyWgosGv9ekMoXGy_6CSg?e=ElIg3i)  <br/>
[Model trained on UCF-101 (K=4, p=0.75)](https://1drv.ms/f/s!Amih97wvmSyWgosIEgaDNoDbxRFDYQ?e=PWZnNA)  <br/>

When training in a distributed manner, you must manually divide the `--batch_size` argument by the number of ranks. In lieu of distributed training, you may use `--microbatch 16` (or `--microbatch 1` in extreme memory-limited cases) to reduce memory usage.

The logs and saved models will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp`.

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
python scripts/video_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

Again, this will save results to a logging directory. Samples are saved as a large `npz` file, where `arr_0` in the file is a large batch of samples.

Just like for training, you can run `image_sample.py` through MPI to use multiple GPUs and machines.

You can change the number of sampling steps using the `--timestep_respacing` argument. For example, `--timestep_respacing 250` uses 250 steps to sample.



