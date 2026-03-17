import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import fastmri
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(__file__))

from dataset import CineCMRDataset
from promptmr import PromptMR

# ================= 推理配置 =================
CONFIG = {
    "root_path": r"/root/autodl-tmp/TestSet/AccFactor04",
    "acc_factor": "04",
    "num_adj": 5,
    "frame_sampling": "all",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "./checkpoints/best_model.pth",
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
    "use_crop_eval": True,
    "save_dir": None,
}
# ===========================================


def matlab_round(n):
    return int(n + 0.5) if n > 0 else int(n - 0.5)


def center_crop_hw(x, crop_h, crop_w):
    h, w = x.shape[-2], x.shape[-1]
    h_start = (h - crop_h) // 2
    w_start = (w - crop_w) // 2
    return x[..., h_start:h_start + crop_h, w_start:w_start + crop_w]


def crop_for_cmrx_cine(x):
    crop_h = matlab_round(x.shape[-2] / 3)
    crop_w = matlab_round(x.shape[-1] / 2)
    return center_crop_hw(x, crop_h, crop_w)


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


def evaluate():
    device = torch.device(CONFIG["device"])

    print(f"Loading model: {CONFIG['model_path']}")
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
        use_checkpoint=False,
    ).to(device)

    state = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    ds = CineCMRDataset(
        root_dir=CONFIG["root_path"],
        acc_factor=CONFIG["acc_factor"],
        num_adj=CONFIG["num_adj"],
        is_train=False,
        frame_sampling=CONFIG["frame_sampling"],
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    if CONFIG["save_dir"]:
        os.makedirs(CONFIG["save_dir"], exist_ok=True)

    psnr_list = []
    ssim_list = []

    for idx, batch in tqdm(enumerate(loader), total=len(loader), desc="infer"):
        input_k = batch["input_kspace"].to(device)
        target_k = batch["target_kspace"].to(device)
        mask = batch["mask"].to(device)

        input_k_ri = to_ri(input_k)
        target_k_ri = to_ri(target_k)
        input_k_flat, mask_flat = flatten_adj_coils(input_k_ri, mask)

        pred = model(input_k_flat, mask_flat, None)
        gt = rss_image_from_kspace(target_k_ri)

        pred_np = pred.detach().cpu().numpy()[0]
        gt_np = gt.detach().cpu().numpy()[0]

        max_val = float(gt_np.max()) if float(gt_np.max()) > 0 else 1.0
        if CONFIG["use_crop_eval"]:
            gt_eval = crop_for_cmrx_cine(gt_np)
            pred_eval = crop_for_cmrx_cine(pred_np)
        else:
            gt_eval = gt_np
            pred_eval = pred_np

        p = psnr(gt_eval, pred_eval, data_range=max_val)
        s = ssim(gt_eval, pred_eval, data_range=max_val)
        psnr_list.append(float(p))
        ssim_list.append(float(s))

        if CONFIG["save_dir"] and idx < 50:
            np.save(os.path.join(CONFIG["save_dir"], f"pred_{idx:05d}.npy"), pred_np)
            np.save(os.path.join(CONFIG["save_dir"], f"gt_{idx:05d}.npy"), gt_np)

    if psnr_list:
        print(f"Average PSNR: {np.mean(psnr_list):.4f} dB")
        print(f"Average SSIM: {np.mean(ssim_list):.6f}")
    else:
        print("未计算到指标（可能 GT 为 0 或数据集为空）")


if __name__ == "__main__":
    evaluate()
