import io
import ffmpeg
import IPython.display as ipd

from base64 import b64decode
from scipy.io.wavfile import read as wav_read
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio

from audioseal import AudioSeal
from datasets import load_dataset

from typing import Tuple, Union
from pystoi.stoi import stoi
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

import argparse

def plot_waveform_and_specgram(waveform, sample_rate, title):
    waveform = waveform.squeeze().detach().cpu().numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(time_axis, waveform, linewidth=1)
    ax1.grid(True)
    ax2.specgram(waveform, Fs=sample_rate)

    figure.suptitle(f"{title} - Waveform and specgram")
    plt.show()


def play_audio(waveform, sample_rate):
    if waveform.dim() > 2:
        waveform = waveform.squeeze(0)
    waveform = waveform.detach().cpu().numpy()

    num_channels, *_ = waveform.shape
    if num_channels == 1:
        ipd.display(ipd.Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        ipd.display(ipd.Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")

ArrayLike = Union[torch.Tensor]

def _to_tensor(x: ArrayLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)

def _flatten_audio(x: torch.Tensor) -> torch.Tensor:
    x = _to_tensor(x).detach()

    if x.dim() == 1:          # (T,)
        x = x.unsqueeze(0)    # (1,T)
    elif x.dim() == 2:        # (C,T) or (B,T)
        # Heuristic: if first dim is small (<=8), treat as (C,T) and downmix
        if x.shape[0] <= 8 and x.shape[1] > 8:
            x = x.mean(dim=0, keepdim=True)  # (1,T)
        # else assume already (B,T)
    elif x.dim() == 3:        # (B,C,T)
        x = x.mean(dim=1)     # (B,T)
    else:
        raise ValueError(f"Unsupported audio shape: {tuple(x.shape)}")

    return x.to(torch.float32)

def _match_length(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x = _flatten_audio(x)
    y = _flatten_audio(y)
    T = min(x.shape[-1], y.shape[-1])
    return x[..., :T], y[..., :T]

def pesq_score(x: torch.Tensor, y: torch.Tensor, sr: int) -> torch.Tensor:
    from pesq import pesq

    if sr not in (8000, 16000):
        raise ValueError("PESQ typically requires sr to be 8000 or 16000.")

    mode = "wb" if sr == 16000 else "nb"

    x, y = _match_length(x, y)
    scores = []
    for i in range(x.shape[0]):
        xi = x[i].cpu().numpy()
        yi = y[i].cpu().numpy()
        scores.append(pesq(sr, xi, yi, mode))
    return torch.tensor(scores, dtype=torch.float32)

def ensure_bct(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (1,T)
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (B,1,T)
    if x.dim() != 3:
        raise ValueError(f"Got shape {tuple(x.shape)}")
    return x

class AudioDenoiseAE(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        # Encoder: downsample by 2 x 3 = 8
        self.enc = nn.Sequential(
            nn.Conv1d(1, base, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(base, base, kernel_size=9, stride=2, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(base, base*2, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(base*2, base*2, kernel_size=9, stride=2, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(base*2, base*4, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(base*4, base*4, kernel_size=9, stride=2, padding=4), nn.ReLU(inplace=True),
        )
        self.mid = nn.Sequential(
            nn.Conv1d(base*4, base*4, kernel_size=9, padding=4), nn.ReLU(inplace=True),
        )
        # Decoder: upsample back
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(base*4, base*4, kernel_size=8, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(base*4, base*2, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(base*2, base*2, kernel_size=8, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(base*2, base, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(base, base, kernel_size=8, stride=2, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(base, 1, kernel_size=9, padding=4),
            nn.Tanh(),  # keep output in [-1,1]
        )

    def forward(self, x):
        x = ensure_bct(x)
        z = self.enc(x)
        z = self.mid(z)
        y = self.dec(z)
        # Safety clamp (optional)
        return y.clamp(-1.0, 1.0)

def stft_mag(x, n_fft, hop):
    # x: (B,1,T) -> (B, F, frames)
    x = x.squeeze(1)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   return_complex=True, center=True)
    return X.abs()

def mrstft_loss(x_hat, x_clean):
    # x_hat/x_clean: (B,1,T)
    cfgs = [(256, 64), (512, 128), (1024, 256)]
    loss = 0.0
    for n_fft, hop in cfgs:
        A = stft_mag(x_hat, n_fft, hop)
        B = stft_mag(x_clean, n_fft, hop)
        loss = loss + F.l1_loss(A, B)
    return loss / len(cfgs)

def train_step_with_stft(ae, x_attack, x_clean, opt, alpha=0.5):
    ae.train()
    x_attack = ensure_bct(x_attack)
    x_clean  = ensure_bct(x_clean)

    opt.zero_grad(set_to_none=True)
    x_hat = ae(x_attack)

    loss_l1 = F.l1_loss(x_hat, x_clean)
    loss_stft = mrstft_loss(x_hat, x_clean)

    loss = loss_l1 + alpha * loss_stft
    loss.backward()
    opt.step()

    return {"loss": float(loss.item()), "l1": float(loss_l1.item()), "stft": float(loss_stft.item())}

def si_sdr(x_hat, x):
    # Match lengths before computing SI-SDR
    x_hat, x = _match_length(x_hat, x)
    
    # _match_length already flattens to (B, T), so no need to squeeze
    eps = 1e-8

    dot = torch.sum(x_hat * x, dim=1, keepdim=True)
    norm = torch.sum(x * x, dim=1, keepdim=True) + eps

    s_target = dot / norm * x
    e_noise = x_hat - s_target

    ratio = torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps)
    return 10 * torch.log10(ratio + eps)

def save_wav(path, audio_tensor, sample_rate=16000):

    # If batched, save first item
    if audio_tensor.dim() == 3:
        audio_tensor = audio_tensor[0]

    # Ensure (1,T)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    # Clamp to valid range
    audio_tensor = audio_tensor.clamp(-1.0, 1.0)

    torchaudio.save(path, audio_tensor.cpu(), sample_rate)

def collate_audio(ds_examples):
    audios = []
    srs = []
    metas = []

    for ex in ds_examples:
        try:
            a = ex["audio"]["array"]
        except:
            continue
        sr = ex["audio"]["sampling_rate"]
        x = torch.as_tensor(a, dtype=torch.float32)

        # mono -> (1,T)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 2:
            x = x.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unexpected audio shape: {tuple(x.shape)}")

        audios.append(x)   # (1,T)
        srs.append(sr)
        metas.append(ex)

    lengths = torch.tensor([x.shape[-1] for x in audios], dtype=torch.long)
    Tmax = int(lengths.max().item())

    batch = torch.zeros((len(audios), 1, Tmax), dtype=torch.float32)
    for i, x in enumerate(audios):
        batch[i, :, :x.shape[-1]] = x

    return batch, lengths, srs, metas

def train_step_epoch(ae, watermark_model, detector, batch_clean, lengths, srs, opt,
                     alpha_wm=1.0, w_stft=0.5, w_sisdr=0.01):
    """
    batch_clean: (B,1,Tmax) float
    lengths: (B,)
    srs: list[int] length B (ideally all same)
    """
    device = next(ae.parameters()).device
    ae.train()

    # Move tensors
    x_clean = batch_clean.to(device)

    # Watermark model & detector should already be on device and in eval mode
    # Generate watermarked batch (no grad)
    with torch.no_grad():
        # NOTE: AudioSeal expects per-sample sample_rate; batching assumes same sr.
        # If srs vary, you MUST resample to a fixed sr before batching.
        sr0 = srs[0]
        if any(sr != sr0 for sr in srs):
            raise ValueError(f"Mixed sample rates in batch: {set(srs)}. Resample or bucket by sr.")
        x_wm = watermark_model(x_clean, sample_rate=sr0, alpha=alpha_wm)

    # AE forward (grad)
    opt.zero_grad(set_to_none=True)
    x_hat = ae(x_wm)

    # Mask padding for waveform losses (important!)
    # Build mask: (B,1,T)
    B, _, T = x_clean.shape
    t = torch.arange(T, device=device).view(1, 1, T)
    mask = (t < lengths.view(B, 1, 1).to(device)).float()

    # Waveform L1 on valid region only
    loss_l1 = (torch.abs(x_hat - x_clean) * mask).sum() / (mask.sum() + 1e-8)

    # MRSTFT / SI-SDR: easiest is to run on cropped per-sample segments.
    # For speed, you can just use the first N samples (or implement masked STFT).
    # Here: crop to min length in batch to avoid padding artifacts:
    Lmin = int(lengths.min().item())
    x_hat_c = x_hat[:, :, :Lmin]
    x_clean_c = x_clean[:, :, :Lmin]

    loss_stft = mrstft_loss(x_hat_c, x_clean_c)
    loss_sisdr = -si_sdr(x_hat_c, x_clean_c).mean()  # negative to maximize SI-SDR

    loss = loss_l1 + w_stft * loss_stft + w_sisdr * loss_sisdr
    loss.backward()
    opt.step()

    return {
        "loss": float(loss.item()),
        "l1": float(loss_l1.item()),
        "stft": float(loss_stft.item()),
        "si_sdr_loss": float(loss_sisdr.item()),
    }

def find_latest_valid_checkpoint(checkpoint_dir):
    """Find and load the latest valid checkpoint."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all checkpoint files
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_ep") and f.endswith(".pt")]
    if not ckpt_files:
        return None
    
    # Sort by epoch and step (extract from filename)
    def parse_ckpt_name(fname):
        # Format: checkpoint_ep{epoch}_step{step}.pt
        try:
            parts = fname.replace(".pt", "").split("_")
            epoch = int(parts[1].replace("ep", ""))
            step = int(parts[2].replace("step", ""))
            return (epoch, step)
        except:
            return (-1, -1)
    
    ckpt_files.sort(key=parse_ckpt_name, reverse=True)
    
    # Try to load checkpoints in order (most recent first)
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        try:
            # Use weights_only=False since these are our own trusted checkpoints
            # and they contain numpy RNG states
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Verify checkpoint has required keys
            required_keys = ['epoch', 'step', 'model_state_dict', 'optimizer_state_dict']
            if all(k in checkpoint for k in required_keys):
                print(f"Successfully loaded checkpoint: {ckpt_file}")
                return checkpoint
            else:
                print(f"Checkpoint {ckpt_file} missing required keys, trying next...")
        except Exception as e:
            print(f"Failed to load {ckpt_file}: {e}, trying next...")
    
    return None

def find_checkpoint_by_epoch_step(checkpoint_dir, epoch=None, step=None):
    """Find checkpoint by specific epoch and/or step."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_ep") and f.endswith(".pt")]
    if not ckpt_files:
        return None
    
    for ckpt_file in ckpt_files:
        try:
            parts = ckpt_file.replace(".pt", "").split("_")
            ep = int(parts[1].replace("ep", ""))
            st = int(parts[2].replace("step", ""))
            if (epoch is None or ep == epoch) and (step is None or st == step):
                ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                print(f"Found checkpoint: {ckpt_file}")
                return checkpoint
        except:
            continue
    
    print(f"No checkpoint found for epoch={epoch} step={step}")
    return None

def save_checkpoint(checkpoint_dir, epoch, step, ae, opt, rng_state):
    """Save checkpoint with unique name."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_ep{epoch}_step{step}.pt")
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': ae.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'rng_state': rng_state,
    }
    
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")
    return ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode-train", action="store_true", help="Run training loop", default=False)
    parser.add_argument("--mode-eval-epoch", type=int, help="Run evaluation on a specific epoch checkpoint", default=None)
    parser.add_argument("--mode-eval-step", type=int, help="Run evaluation on a specific step checkpoint", default=None)
    parser.add_argument("--mode-eval-count", type=int, help="Number of evaluation samples to run (default: 64)", default=None)
    parser.add_argument("--eval-elevenlabs", action="store_true", help="Run evaluation on ElevenLabs watermarked audio", default=True)
    args = parser.parse_args()

    HF_TOKEN = ""
    ds_train = load_dataset("MLCommons/peoples_speech", "clean", split="train", streaming=True, token=HF_TOKEN)
    print(ds_train)

    model = AudioSeal.load_generator("audioseal_wm_16bits")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 64
    num_epochs = 10
    log_every = 10
    save_examples = 64 # 64 per epoch
    checkpoint_dir = "/local/homework_cs358800_winter_2026/raniayu/project/checkpoints"

    ae = AudioDenoiseAE(base=32).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    
    if args.mode_train:
        # Try to load checkpoint
        start_epoch = 1
        start_step = 0
        checkpoint = find_latest_valid_checkpoint(checkpoint_dir)
        if checkpoint is not None:
            ae.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            
            # Restore RNG state for reproducibility
            if 'rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['rng_state']['torch_rng_state'])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(checkpoint['rng_state']['cuda_rng_state'])
                random.setstate(checkpoint['rng_state']['python_rng_state'])
                np.random.set_state(checkpoint['rng_state']['numpy_rng_state'])
            
            # If we completed evaluation for this epoch, move to next epoch
            if start_step >= 2000:
                start_epoch += 1
                start_step = 0
            
            print(f"Resuming from epoch {start_epoch}, step {start_step}")
        else:
            print("No valid checkpoint found, starting from scratch")

        # Put watermark model + detector on device ONCE
        model = model.to(device).eval()
        detector = detector.to(device).eval()
        for p in detector.parameters():
            p.requires_grad_(False)

        # ae.eval()

        # import torchaudio
        # watermarked_audio, sr = torchaudio.load("/local/homework_cs358800_winter_2026/raniayu/project/audios/output.mp3")
        # print(sr)

        # # AudioSegment.from_mp3(f'/local/homework_cs358800_winter_2026/raniayu/project/audios/ElevenLabs_2026-01-20T23_02_30_Rachel_pre_sp100_s50_sb75_se0_b_m2.mp3').set_frame_rate(16000).export(
        # #     f'/local/homework_cs358800_winter_2026/raniayu/project/output/purified_downsample.wav', 
        # #     format='wav', 
        # #     bitrate='128k'
        # # )

        # # watermarked_audio, sr_mod = torchaudio.load("/local/homework_cs358800_winter_2026/raniayu/project/audios/ElevenLabs_2026-01-20T23_02_30_Rachel_pre_sp100_s50_sb75_se0_b_m2.mp3")
        # # print(sr_mod)

        # with torch.no_grad():
        #     x_clean = watermarked_audio.unsqueeze(0).to(device)  # (1,C,T)
        #     if x_clean.shape[1] != 1:
        #         x_clean = x_clean.mean(dim=1, keepdim=True)      # mono -> (1,1,T)
        #     x_wm = model(x_clean, sample_rate=sr, alpha=1.0)
        #     x_hat = ae(x_wm)

        #     result_wm, msg_wm = detector.detect_watermark(
        #         x_wm, sample_rate=sr, message_threshold=0.5
        #     )

        #     # Detect on original
        #     result_orig, msg_orig = detector.detect_watermark(
        #         x_clean, sample_rate=sr, message_threshold=0.5
        #     )
        #     # Detect on purified
        #     result_pur, msg_pur = detector.detect_watermark(
        #         x_hat, sample_rate=sr, message_threshold=0.5
        #     )

        #     print(f"Original: detected={result_orig} msg_prob={msg_orig}")
        #     print(f"Watermarked: detected={result_wm} msg_prob={msg_wm}")
        #     print(f"Purified: detected={result_pur} msg_prob={msg_pur}")

        #     # save purified
        #     save_wav("/local/homework_cs358800_winter_2026/raniayu/project/output/purified_example.wav", x_hat[0], sample_rate=sr)

        loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_audio,
        )

        for epoch in range(start_epoch, num_epochs + 1):
            running = {"loss": 0.0, "l1": 0.0, "stft": 0.0, "si_sdr_loss": 0.0}
            n_steps = 0
            
            # Determine where to start counting in this epoch
            # If resuming mid-epoch, start from the next step after checkpoint
            step_counter = start_step if epoch == start_epoch else 0

            for batch_idx, (batch_clean, lengths, srs, metas) in enumerate(loader):
                step_counter += 1
                
                if epoch == start_epoch and step_counter <= start_step:
                    continue
                
                stats = train_step_epoch(
                    ae, model, detector,
                    batch_clean, lengths, srs,
                    opt,
                    alpha_wm=1.0,
                    w_stft=0.5,
                    w_sisdr=0.01,
                )

                for k in running:
                    running[k] += stats[k]
                n_steps += 1

                if step_counter % log_every == 0:
                    avg = {k: running[k] / n_steps for k in running}
                    print(f"epoch {epoch} step {step_counter}: {avg}")
                
                # Save checkpoint every 100 steps
                if step_counter % 100 == 0:
                    rng_state = {
                        'torch_rng_state': torch.get_rng_state(),
                        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'python_rng_state': random.getstate(),
                        'numpy_rng_state': np.random.get_state(),
                    }
                    save_checkpoint(checkpoint_dir, epoch, step_counter, ae, opt, rng_state)
                
                if step_counter >= 2000:
                    break

            # Fetch a fresh batch for evaluation (unseen data)
            print(f"[epoch {epoch}] Fetching unseen evaluation batch...")
            try:
                eval_batch_clean, eval_lengths, eval_srs, eval_metas = next(iter(loader))
            except StopIteration:
                print("No evaluation data available, skipping evaluation")
                continue

            # Evaluate on up to 10 samples from the evaluation batch
            num_eval_samples = min(batch_size, eval_batch_clean.size(0))
            eval_batch_clean = eval_batch_clean[:num_eval_samples]
            eval_lengths = eval_lengths[:num_eval_samples]
            eval_srs = eval_srs[:num_eval_samples]
            eval_metas = eval_metas[:num_eval_samples]

            ae.eval()
            with torch.no_grad():
                sr0 = eval_srs[0]
                x_clean = eval_batch_clean.to(device)
                x_wm = model(x_clean, sample_rate=sr0, alpha=1.0)
                x_hat = ae(x_wm)

                result_wm, msg_wm = detector.detect_watermark(
                    x_wm, sample_rate=sr0, message_threshold=0.5
                )

                # Detect on original
                result_orig, msg_orig = detector.detect_watermark(
                    x_clean, sample_rate=sr0, message_threshold=0.5
                )
                # Detect on purified
                result_pur, msg_pur = detector.detect_watermark(
                    x_hat, sample_rate=sr0, message_threshold=0.5
                )

                print(f"Original: detected={result_orig} msg_prob={msg_orig}")
                print(f"Watermarked: detected={result_wm} msg_prob={msg_wm}")
                print(f"Purified: detected={result_pur} msg_prob={msg_pur}")

            save_dir = "/local/homework_cs358800_winter_2026/raniayu/project/output"
            save_path = os.path.join(save_dir, "audio_denoise_ae.pt")
            torch.save(ae.state_dict(), save_path)
            print("Model saved to:", save_path)

            # Save and evaluate all samples (up to 10)
            eval_indices = list(range(num_eval_samples))
            for i in eval_indices:
                L = int(eval_lengths[i].item())
                ex = eval_metas[i]

                orig = x_clean[i:i+1, :, :L].detach().cpu()
                wm   = x_wm[i:i+1, :, :L].detach().cpu()
                pur  = x_hat[i:i+1, :, :L].detach().cpu()

                save_wav(f"/local/homework_cs358800_winter_2026/raniayu/project/output/new-ep{epoch}-orig-{i}.wav", orig, sample_rate=sr0)
                save_wav(f"/local/homework_cs358800_winter_2026/raniayu/project/output/new-ep{epoch}-wm-{i}.wav", wm, sample_rate=sr0)
                save_wav(f"/local/homework_cs358800_winter_2026/raniayu/project/output/new-ep{epoch}-pur-{i}.wav", pur, sample_rate=sr0)

                # PESQ expects (B,T) typically
                try:
                    pesq_val = pesq_score(orig.squeeze(1), pur.squeeze(1), sr0).mean().item()
                except Exception as e:
                    pesq_val = None
                    print("PESQ failed:", e)

                # si-sdr
                try:
                    sisdr_val = si_sdr(pur, orig).item()
                except Exception as e:
                    sisdr_val = None
                    print("SI-SDR failed:", e)

                print(f"[epoch {epoch}] sample {i} id={ex.get('id', None)} pesq={pesq_val} si-sdr={sisdr_val}")
            
            # Save checkpoint after evaluation completes
            rng_state = {
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
            }
            save_checkpoint(checkpoint_dir, epoch, 2000, ae, opt, rng_state)
            print(f"Epoch {epoch} completed.")
    
    if args.mode_eval_epoch is not None or args.mode_eval_step is not None:
        epoch = args.mode_eval_epoch if args.mode_eval_epoch is not None else 0
        step = args.mode_eval_step if args.mode_eval_step is not None else 0
        
        # Get the specified checkpoint
        checkpoint = find_checkpoint_by_epoch_step(checkpoint_dir, epoch=args.mode_eval_epoch, step=args.mode_eval_step)
        if checkpoint is None:
            print("No valid checkpoint found for evaluation, exiting.")
            exit(1)
        # Get 2 random batches from the data loader (doesn't matter train or eval split)
        ae.load_state_dict(checkpoint['model_state_dict'])

        # Put watermark model + detector on device ONCE
        model = model.to(device).eval()
        detector = detector.to(device).eval()
        for p in detector.parameters():
            p.requires_grad_(False)

        random_loader = DataLoader(
            ds_train.shuffle(seed=42),  # Shuffle with fixed seed for reproducibility
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_audio,
        )

        # Fetch a fresh batch for evaluation (unseen data)
        print(f"[epoch {epoch}] Fetching unseen evaluation batch...")
        try:
            eval_batch_clean, eval_lengths, eval_srs, eval_metas = next(iter(random_loader))
        except StopIteration:
            print("No evaluation data available, skipping evaluation")
            exit(1)

        # Evaluate on up to 10 samples from the evaluation batch
        num_eval_samples = min(batch_size, eval_batch_clean.size(0))
        eval_batch_clean = eval_batch_clean[:num_eval_samples]
        eval_lengths = eval_lengths[:num_eval_samples]
        eval_srs = eval_srs[:num_eval_samples]
        eval_metas = eval_metas[:num_eval_samples]

        # Stats variables
        avg_pesq = 0.0
        avg_sisdr = 0.0
        count_success_purify = 0
        avg_success_purify = 0.0

        ae.eval()
        with torch.no_grad():
            sr0 = eval_srs[0]
            x_clean = eval_batch_clean.to(device)
            x_wm = model(x_clean, sample_rate=sr0, alpha=1.0)
            x_hat = ae(x_wm)

            result_wm, msg_wm = detector.detect_watermark(
                x_wm, sample_rate=sr0, message_threshold=0.5
            )

            # Detect on original
            result_orig, msg_orig = detector.detect_watermark(
                x_clean, sample_rate=sr0, message_threshold=0.5
            )
            # Detect on purified
            result_pur, msg_pur = detector.detect_watermark(
                x_hat, sample_rate=sr0, message_threshold=0.5
            )

            print(f"Original: detected={result_orig}") #  msg_prob={msg_orig}
            print(f"Watermarked: detected={result_wm}") #  msg_prob={msg_wm}
            print(f"Purified: detected={result_pur}") #  msg_prob={msg_pur}

            count_success_purify = (result_pur < 0.5).sum().item()
            avg_success_purify = result_pur.mean().item()
            
            print(f"Purification success count: {count_success_purify}/{num_eval_samples} avg msg_prob={avg_success_purify}")

            # get pesq and si-sdr for each sample
            for i in range(num_eval_samples):
                L = int(eval_lengths[i].item())
                ex = eval_metas[i]

                orig = x_clean[i:i+1, :, :L].detach().cpu()
                pur  = x_hat[i:i+1, :, :L].detach().cpu()

                # PESQ expects (B,T) typically
                try:
                    pesq_val = pesq_score(orig.squeeze(1), pur.squeeze(1), sr0).mean().item()
                except Exception as e:
                    pesq_val = None
                    print("PESQ failed:", e)

                # si-sdr
                try:
                    sisdr_val = si_sdr(pur, orig).item()
                except Exception as e:
                    sisdr_val = None
                    print("SI-SDR failed:", e)

                if pesq_val is not None:
                    avg_pesq += pesq_val
                if sisdr_val is not None:
                    avg_sisdr += sisdr_val

                print(f"Sample {i} id={ex.get('id', None)} pesq={pesq_val} si-sdr={sisdr_val}")
            
            print(f"Average PESQ: {avg_pesq / num_eval_samples:.4f}")
            print(f"Average SI-SDR: {avg_sisdr / num_eval_samples:.4f}")
    
    if args.eval_elevenlabs:

        # Get the specified checkpoint
        checkpoint = find_checkpoint_by_epoch_step(checkpoint_dir, epoch=6, step=2000)
        if checkpoint is None:
            print("No valid checkpoint found for evaluation, exiting.")
            exit(1)
        # Get 2 random batches from the data loader (doesn't matter train or eval split)
        ae.load_state_dict(checkpoint['model_state_dict'])

        # Put watermark model + detector on device ONCE
        model = model.to(device).eval()
        detector = detector.to(device).eval()
        for p in detector.parameters():
            p.requires_grad_(False)

        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-02-03T19_13_02_Amelia - Enthusiastic and Expressive_pvc_sp100_s50_sb48_se0_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-01-20T23_02_30_Rachel_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-02-03T19_04_43_Adam_pvc_sp110_s65_sb55_se35_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-02-03T19_10_36_Quentin - Narrator and Educator_pvc_sp105_s34_sb100_se0_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_33_16_Lily - Velvety Actress_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_35_20_Eric - Smooth, Trustworthy_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_36_13_Sarah - Mature, Reassuring, Confident_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_37_11_River - Relaxed, Neutral, Informative_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_38_11_Matilda - Knowledgable, Professional_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_38_55_Roger - Laid-Back, Casual, Resonant_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_39_27_Brian - Deep, Resonant and Comforting_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_40_40_Bill - Wise, Mature, Balanced_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_41_25_Charlie - Deep, Confident, Energetic_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_44_11_Harry - Fierce Warrior_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_45_05_Jessica - Playful, Bright, Warm_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_2026-03-02T17_46_52_Brian - Deep, Resonant and Comforting_pre_sp100_s50_sb75_se0_b_m2.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_20260225T16_51_52_Japanese.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_20260225T17_01_27_Boston_English.mp3"
        # path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_20260225T17_14_16_British_English.mp3"
        path = "/local/homework_cs358800_winter_2026/raniayu/project/audios/16000-ElevenLabs_20260225T17_19_03_EdwardBritishDarkSeductiveLow_pvc_sp100_s50_sb75_se0_b_m2.mp3"

        watermarked_audio, sr = torchaudio.load(path)
        print(sr)

        with torch.no_grad():
            x_clean = watermarked_audio.unsqueeze(0).to(device)  # (1,C,T)
            if x_clean.shape[1] != 1:
                x_clean = x_clean.mean(dim=1, keepdim=True)      # mono -> (1,1,T)
            x_wm = model(x_clean, sample_rate=sr, alpha=1.0)
            x_hat = ae(x_wm)
            x_hat_direct = ae(x_clean)

            result_wm, msg_wm = detector.detect_watermark(
                x_wm, sample_rate=sr, message_threshold=0.5
            )

            # Detect on original
            result_orig, msg_orig = detector.detect_watermark(
                x_clean, sample_rate=sr, message_threshold=0.5
            )
            # Detect on purified
            result_pur, msg_pur = detector.detect_watermark(
                x_hat, sample_rate=sr, message_threshold=0.5
            )
            # Detect on purified elevenlabs
            result_pur_direct, msg_pur_direct = detector.detect_watermark(
                x_hat_direct, sample_rate=sr, message_threshold=0.5
            )

            print(f"Original: detected={result_orig} msg_prob={msg_orig}")
            print(f"Watermarked: detected={result_wm} msg_prob={msg_wm}")
            print(f"Purified: detected={result_pur} msg_prob={msg_pur}")
            print(f"Purified direct: detected={result_pur_direct} msg_prob={msg_pur_direct}")

            # get pesq and si-sdr for purified
            try:
                pesq_val_xhat = pesq_score(x_clean.squeeze(1), x_hat.squeeze(1), sr).mean().item()
                pesq_val_xhat_direct = pesq_score(x_clean.squeeze(1), x_hat_direct.squeeze(1), sr).mean().item()
            except Exception as e:
                pesq_val = None
                print("PESQ failed:", e)
            try:
                sisdr_val_xhat = si_sdr(x_hat, x_clean).item()
                sisdr_val_direct = si_sdr(x_hat_direct, x_clean).item()
            except Exception as e:
                sisdr_val = None
                print("SI-SDR failed:", e)
            print(f"Purified PESQ: {pesq_val_xhat:.4f} SI-SDR: {sisdr_val_xhat:.4f}")
            print(f"Purified direct PESQ: {pesq_val_xhat_direct:.4f} SI-SDR: {sisdr_val_direct:.4f}")

            # save purified
            basename = os.path.basename(path)
            save_wav(f"/local/homework_cs358800_winter_2026/raniayu/project/audios/purified_{basename}.wav", x_hat[0], sample_rate=sr)
            save_wav(f"/local/homework_cs358800_winter_2026/raniayu/project/audios/purified_direct_{basename}.wav", x_hat_direct[0], sample_rate=sr)