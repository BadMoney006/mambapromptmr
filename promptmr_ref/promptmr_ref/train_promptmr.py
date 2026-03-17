import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import fastmri
from fastmri.losses import SSIMLoss

sys.path.insert(0, os.path.dirname(__file__))

from dataset import CineCMRDataset
from promptmr import PromptMR

# ================= 配置区域 =================
CONFIG = {
    "root_path": r"/root/autodl-tmp/Cine/TrainingSet/AccFactor04",
    "acc_factor": "04",
    "num_adj": 5,
    "frame_sampling": "random_k",
    "num_sampled_frames": 3,
    "seed": 42,
    "batch_size": 1,
    "num_epochs": 50,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "num_cascades": 12,
    "n_feat0": 48,
    "feature_dim": [72, 96, 120],
    "prompt_dim": [24, 48, 72],
    "len_prompt": [5, 5, 5],
    "prompt_size": [64, 32, 16],
    "n_enc_cab": [2, 3, 3],
    "n_dec_cab": [2, 2, 3],
    "n_skip_cab": [1, 1, 1],
    "n_bottleneck_cab": 3,
    "no_use_ca": False,
    "use_checkpoint": True,
    "use_amp": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./checkpoints",
    "grad_clip": 0.1,
}
# ===========================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_ri(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x) if x.is_complex() else x


def flatten_adj_coils(kspace_ri: torch.Tensor, mask: torch.Tensor):
    b, t, c, h, w, two = kspace_ri.shape
    kspace_flat = kspace_ri.view(b, t * c, h, w, two)
    mask_flat = mask.view(b, t * c, h, w).unsqueeze(-1).bool()
    return kspace_flat, mask_flat


def rss_image_from_kspace(kspace: torch.Tensor) -> torch.Tensor:
    if kspace.is_complex():
        kspace = torch.view_as_real(kspace)
    img = fastmri.ifft2c(kspace)
    mag = fastmri.complex_abs(img)
    return fastmri.rss(mag, dim=1)


def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    print("Loading training data...")
    train_dataset = CineCMRDataset(
        root_dir=CONFIG["root_path"],
        acc_factor=CONFIG["acc_factor"],
        num_adj=CONFIG["num_adj"],
        is_train=True,
        frame_sampling=CONFIG["frame_sampling"],
        num_sampled_frames=CONFIG["num_sampled_frames"],
        seed=CONFIG["seed"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    print("Initializing PromptMR...")
    model = PromptMR(
        num_cascades=CONFIG["num_cascades"],
        num_adj_slices=CONFIG["num_adj"],
        n_feat0=CONFIG["n_feat0"],
        feature_dim=CONFIG["feature_dim"],
        prompt_dim=CONFIG["prompt_dim"],
        len_prompt=CONFIG["len_prompt"],
        prompt_size=CONFIG["prompt_size"],
        n_enc_cab=CONFIG["n_enc_cab"],
        n_dec_cab=CONFIG["n_dec_cab"],
        n_skip_cab=CONFIG["n_skip_cab"],
        n_bottleneck_cab=CONFIG["n_bottleneck_cab"],
        no_use_ca=CONFIG["no_use_ca"],
        use_checkpoint=CONFIG["use_checkpoint"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(CONFIG["num_epochs"] - 1, 1), gamma=0.1
    )
    criterion_ssim = SSIMLoss().to(device)
    scaler = GradScaler(enabled=CONFIG["use_amp"])

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        for batch in progress:
            input_k = batch["input_kspace"].to(device)
            target_k = batch["target_kspace"].to(device)
            mask = batch["mask"].to(device)

            input_k_ri = to_ri(input_k)
            target_k_ri = to_ri(target_k)
            input_k_flat, mask_flat = flatten_adj_coils(input_k_ri, mask)

            target_img = rss_image_from_kspace(target_k_ri)

            optimizer.zero_grad()
            with autocast(enabled=CONFIG["use_amp"]):
                output_center = model(input_k_flat, mask_flat, None)

            with autocast(enabled=False):
                output_center_f = output_center.float()
                target_img_f = target_img.float()
                max_val = target_img_f.view(target_img_f.shape[0], -1).max(dim=1)[0].clamp_min(1e-6)
                loss = criterion_ssim(
                    output_center_f.unsqueeze(1), target_img_f.unsqueeze(1), data_range=max_val
                )

            if CONFIG["use_amp"]:
                scaler.scale(loss).backward()
                if CONFIG["grad_clip"] and CONFIG["grad_clip"] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if CONFIG["grad_clip"] and CONFIG["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
                optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.6f}")

        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "latest_model.pth"))
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], f"epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    train()
