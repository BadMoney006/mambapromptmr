import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import fastmri
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dataset import CineCMRDataset
from mamba_prompt_mr_model import MambaPromptMR

# ================= 单样本配置 =================
CONFIG = {
    "root_path": r"/root/autodl-tmp/",
    "model_path": "/root/autodl-tmp/",
    "acc_factor": "04",
    "num_adj": 5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sample_index": 0,  # 数据集中第几个样本
    "use_crop_eval": True,
    "save_path": "./one_slice_result.png",
    "save_dir": "./one_slice_parts",
}
# ===========================================


def matlab_round(n):
    if n > 0:
        return int(n + 0.5)
    else:
        return int(n - 0.5)


def center_crop_hw(x, crop_h, crop_w):
    h, w = x.shape[-2], x.shape[-1]
    h_start = (h - crop_h) // 2
    w_start = (w - crop_w) // 2
    return x[..., h_start:h_start + crop_h, w_start:w_start + crop_w]


def crop_for_cmrx_cine(x):
    crop_h = matlab_round(x.shape[-2] / 3)
    crop_w = matlab_round(x.shape[-1] / 2)
    return center_crop_hw(x, crop_h, crop_w)


def main():
    device = torch.device(CONFIG["device"])

    # 1) Load model
    model = MambaPromptMR(
        num_cascades=8,
        num_adj_slices=CONFIG["num_adj"],
        n_feat0=48,
        img_size=256,
        use_checkpoint=False,
    ).to(device)
    ckpt = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    # 2) Load one sample
    dataset = CineCMRDataset(
        root_dir=CONFIG["root_path"],
        acc_factor=CONFIG["acc_factor"],
        num_adj=CONFIG["num_adj"],
        is_train=False,
    )
    sample = dataset[CONFIG["sample_index"]]
    input_k = sample["input_kspace"].unsqueeze(0).to(device)  # (1, T, C, H, W)
    target_k = sample["target_kspace"].unsqueeze(0).to(device)  # (1, C, H, W)
    mask = sample["mask"].unsqueeze(0).to(device)  # (1, T, C, H, W)

    if input_k.is_complex():
        input_k = torch.view_as_real(input_k)
    if target_k.is_complex():
        target_k = torch.view_as_real(target_k)

    # 3) Zero-filled (center frame)
    center_idx = CONFIG["num_adj"] // 2
    input_k_center = input_k[:, center_idx, ...]  # (1, C, H, W, 2)
    zf_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(input_k_center)), dim=1)

    # 4) Reconstruct
    with torch.no_grad():
        recon_seq = model(input_k, mask)
    recon_img = recon_seq[:, center_idx, ...]  # (1, H, W)

    # 5) Ground truth
    gt_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(target_k)), dim=1)

    # 6) Metric
    zf_np = zf_img.cpu().numpy()[0]
    recon_np = recon_img.cpu().numpy()[0]
    gt_np = gt_img.cpu().numpy()[0]

    max_val = gt_np.max()
    if CONFIG["use_crop_eval"]:
        gt_eval = crop_for_cmrx_cine(gt_np)
        recon_eval = crop_for_cmrx_cine(recon_np)
        zf_eval = crop_for_cmrx_cine(zf_np)
    else:
        gt_eval = gt_np
        recon_eval = recon_np
        zf_eval = zf_np

    recon_psnr = psnr(gt_eval, recon_eval, data_range=max_val)
    recon_ssim = ssim(gt_eval, recon_eval, data_range=max_val)
    zf_psnr = psnr(gt_eval, zf_eval, data_range=max_val)
    zf_ssim = ssim(gt_eval, zf_eval, data_range=max_val)

    recon_psnr_full = psnr(gt_np, recon_np, data_range=max_val)
    recon_ssim_full = ssim(gt_np, recon_np, data_range=max_val)
    zf_psnr_full = psnr(gt_np, zf_np, data_range=max_val)
    zf_ssim_full = ssim(gt_np, zf_np, data_range=max_val)

    print("---- Metrics ----")
    print(f"[Crop] ZF    PSNR: {zf_psnr:.2f} dB, SSIM: {zf_ssim:.4f}")
    print(f"[Crop] Recon PSNR: {recon_psnr:.2f} dB, SSIM: {recon_ssim:.4f}")
    print(f"[Full] ZF    PSNR: {zf_psnr_full:.2f} dB, SSIM: {zf_ssim_full:.4f}")
    print(f"[Full] Recon PSNR: {recon_psnr_full:.2f} dB, SSIM: {recon_ssim_full:.4f}")

    # 7) Error map (full image)
    err_map = np.abs(gt_np - recon_np)

    # 8) Save visualization
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(zf_np, cmap="gray")
    ax[0].set_title("Zero-Filled")
    ax[0].axis("off")

    ax[1].imshow(recon_np, cmap="gray")
    ax[1].set_title(f"Recon\nPSNR: {recon_psnr:.2f}, SSIM: {recon_ssim:.4f}")
    ax[1].axis("off")

    ax[2].imshow(gt_np, cmap="gray")
    ax[2].set_title("Ground Truth")
    ax[2].axis("off")

    im_err = ax[3].imshow(err_map, cmap="jet", vmin=0, vmax=err_map.max())
    ax[3].set_title("Error Map")
    ax[3].axis("off")
    plt.colorbar(im_err, ax=ax[3], fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(CONFIG["save_path"]) or ".", exist_ok=True)
    plt.savefig(CONFIG["save_path"], bbox_inches="tight", dpi=150)
    print(f"Saved: {CONFIG['save_path']}")

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    plt.imsave(os.path.join(CONFIG["save_dir"], "zero_filled.png"), zf_np, cmap="gray")
    plt.imsave(os.path.join(CONFIG["save_dir"], "reconstruction.png"), recon_np, cmap="gray")
    plt.imsave(os.path.join(CONFIG["save_dir"], "ground_truth.png"), gt_np, cmap="gray")
    plt.imsave(os.path.join(CONFIG["save_dir"], "error_map.png"), err_map, cmap="jet")
    print(f"Saved parts to: {CONFIG['save_dir']}")


if __name__ == "__main__":
    main()
