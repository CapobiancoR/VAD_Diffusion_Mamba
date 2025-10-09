import copy
import functools
import os

import os
import shutil
from typing import Union

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

#spostato su run_train.sh
#os.environ["DIFFUSION_BLOB_LOGDIR"] =  "/media/pinas/riccardo/avenue" # "/media/pinas/riccardo/STC_check" # "/home/riccardo/checkpoints" # "/media/splash/simone/VAD_Mamba/log_marzo" #"/media/splash/simone/VAD_Mamba/log_febbraio" # "/media/splash/simone/VAD_Mamba/log_dicembre" contiene i checkpoint prima che arrivasse @Riccardo

print("Checkpoint directory (DIFFUSION_BLOB_LOGDIR):", os.environ.get("DIFFUSION_BLOB_LOGDIR"))
print("Logger directory (logger.get_dir()):", logger.get_dir())

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.d
INITIAL_LOG_LOSS_SCALE = 20.0

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
    logger.log('Removing the following items:', to_delete) 
    for path in to_delete:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            # print(f"Rimosso: {path}")
            #print(f"Removed: {path}") 
            logger.log(f"Removed: {path}")  # Log the removal
        except Exception as e:
            # Qui potresti voler loggare l'errore o rilanciarlo
            print(f"Errore rimuovendo '{path}': {e}")
            logger.log(f"Error removing '{path}': {e}")  # Log the error

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        clip=1,
        anneal_type=None,
        steps_drop=None,
        drop=None,
        decay=None,
        max_num_mask_frames=4,
        mask_range=None, 
        uncondition_rate=True,
        exclude_conditional=True,
        wandb_name=None,
    ):
        self.wandb_name = wandb_name or (None, None, None)  # Default to (None, None, None) if not provided
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.accumulation_steps = batch_size / microbatch if microbatch > 0 else 1
        self.lr = lr
        self.current_lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.anneal_type = anneal_type
        if self.anneal_type == 'linear':
            assert lr_anneal_steps != 0 
            self.lr_anneal_steps = lr_anneal_steps
        if self.anneal_type == 'step':
            assert steps_drop != 0
            assert drop != 0
            self.steps_drop = steps_drop
            self.drop = drop
        if self.anneal_type == 'time_based':
            assert decay != 0
            self.decay = decay

        self.clip = clip
        self.max_num_mask_frames = max_num_mask_frames
        self.mask_range = mask_range
        self.uncondition_rate = uncondition_rate
        self.exclude_conditional = exclude_conditional

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        logger.log(f"global batch size = {self.global_batch}")

        self.model_params = list(self.model.parameters()) # 360
        self.master_params = self.model_params #@simone Originariamente era questo

        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            logger.log(f"world_size: {dist.get_world_size()}")
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        wandb.init(project='RaMViD_', entity='pinlab-sapienza',
                   config={
                "learning_rate": lr,
                "batch_size": batch_size,
                "ema_rate": ema_rate,
                "use_fp16": use_fp16,
            },
            name='TRAIN_RaMViD_'+str(wandb_name[0].split("/")[3])+"_"+str(wandb_name[1].split("/")[-1])
        )
        wandb.watch(self.model)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        name = self.wandb_name[0].split("/")[3]
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint,name)

            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )


        dist_util.sync_params(self.model.parameters())
        # Anche qui, master_params e self.model.named_parameters sono gli stessi

    def _load_ema_parameters(self, rate):
        #print(f"Prima di deepcopy, self.master_params sono {len(self.master_params)}")
        ema_params = copy.deepcopy(self.master_params)
        #print(f"Dopo deepcopy, self.master_params sono {len(self.master_params)}")
        name = self.wandb_name[0].split("/")[3]
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate,name)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        name = self.wandb_name[0].split("/")[3]
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"{name}_opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_opt_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )

            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params) #@simone Originariamente era così...
        #self.master_params = make_master_params(list(self.model.parameters())) # @simone l'ho reso così per compatibilità con MambAttention
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            self.current_lr
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                #print(f"Riga 212: Mismatch between master_params ({len(self.master_params)}) and self.model.named_parameters ({len(list(self.model.named_parameters()))})") # @simone aggiunto per DEBUG
                self.save()
                # Run for a finite amount of time in integration tests. Does access an environment variable
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            #print(f"Riga 220: Mismatch between master_params ({len(self.master_params)}) and self.model.named_parameters ({len(list(self.model.named_parameters()))})")
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.clip:
            th.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.clip)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()
        #print('@simone_e_luca Fine step')

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                max_num_mask_frames=self.max_num_mask_frames,
                mask_range=self.mask_range,
                uncondition_rate=self.uncondition_rate,
                exclude_conditional=self.exclude_conditional,
            )
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            loss = loss / self.accumulation_steps
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        # @simone Queste prime tre righe servono per DEBUG
        for name, param in self.model.named_parameters():
            if param.grad is None:
                #print(f"Il parametro {name} non ha un gradiente.") # @simone aggiunto per DEBUG
                continue
            else:
                #print(f"Il parametro {name} HA un gradiente.") # @simone aggiunto per DEBUG
                continue

        sqsum = 0.0
        for p in self.master_params:
            if p.grad is not None: #@simone Perché Mamba non usa quei model.head weights
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if self.anneal_type is None:
            return
        if self.anneal_type == "linear":
            frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
            lr = self.lr * (1 - frac_done)
        elif self.anneal_type == "step":
            lr = self.lr * self.drop**(np.floor((self.step + self.resume_step)/self.steps_drop))
        elif self.anneal_type == "time_based":
            lr = self.lr / (1 + self.decay * (self.step + self.resume_step))
        else:
            raise ValueError(f"unsupported anneal type: {self.anneal_type}")
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        self.current_lr = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

        wandb.log({
            "step": self.step + self.resume_step,
            "samples": (self.step + self.resume_step + 1) * self.global_batch,
            "learning_rate": self.current_lr,
            "loss_scale": self.lg_loss_scale if self.use_fp16 else None,
            # You can add more metrics here, like loss or gradients.
        })    

    def save(self):
        name = self.wandb_name[0].split("/")[3]
        
        def save_checkpoint(rate, params):
            keep_recent_items(get_blob_logdir(), n=6)
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"{name}_model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"{name}_ema_{rate}_{(self.step+self.resume_step):06d}.pt"

                filepath = bf.join(get_blob_logdir(), filename) #@simone
                print(f"Saving checkpoint to: {filepath}") #@simone
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"{name}_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        # @simone A questo punto già è avvenuto il mismatch
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.model.parameters()), master_params
            )
        state_dict = self.model.state_dict()
        if len(master_params) != len(list(self.model.named_parameters())):
            # DEBUG @simone
            print(f"Careful! Mismatch between master_params ({len(master_params)}) and self.model.named_parameters ({len(list(self.model.named_parameters()))})")
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]

        ## @simone Si riesce ad aggirare l'errore facendo il seguente, ma non penso proprio sia corretto
        # for i, (name, _value) in enumerate(self.model.named_parameters()):
        #     if i < len(master_params):
        #         print(f"Qui è ok: il parametro '{name}' viene aggiunto al dizionario con successo")
        #         state_dict[name] = master_params[i]
        #     else:
        #         print(f"----- Warning: Il parametro '{name}' non ha un corrispondente in master_params.")    
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename,name=""):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    
    split = filename.split(f"{name}_model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate,name=""):
    if main_checkpoint is None:
        return None
    filename = f"{name}_ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        wandb.log({key: values.mean().item()})
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
            wandb.log({f"{key}_q{quartile}": sub_loss})  