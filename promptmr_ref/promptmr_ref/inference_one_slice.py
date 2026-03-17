import os
import sys
import torch
import numpy as np
import fastmri
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(__file__))

from dataset import CineCMRDataset
from promptmr import PromptMR

# ================= 单样本配置 =================
CONFIG = {
    "root_path": r"/root/autodl-tmp/Cine/TrainingSet/AccFactor04",
    "model_path": "/root/autodl-tmp/ckpt/promptmr_best.pth",
    "acc_factor": "04",
    "num_adj": 5,
    "sample_index": 5,
    "frame_sampling": "center",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_crop_eval": True,
    "save_path": "./promptmr_one_slice.png",
    "save_dir": "./promptmr_one_slice_parts",
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


def main():
    device = torch.device(CONFIG["device"])

    model = PromptMR(
        num_cascades=12,
        num_adj_slices=CONFIG["num_adj"],
        n_feat0=48,
        feature_dim=[72, 96, 120],
        prompt_dim=[24, 48, 72],
        len_prompt=[5, 5, 5],
        prompt_size=[64, 32, 16],
        n_enc_cab=[2, 3, 3],
        n_dec_cab=[2, 2, 3],
        n_skip_cab=[1, 1, 1],
        n_bottleneck_cab=3,
        no_use_ca=False,
        use_checkpoint=False,
    ).to(device)

    state = torch.load(CONFIG["model_path"], map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    dataset = CineCMRDataset(
        root_dir=CONFIG["root_path"],
        acc_factor=CONFIG["acc_factor"],
        num_adj=CONFIG["num_adj"],
        is_train=False,
        frame_sampling=CONFIG["frame_sampling"],
    )
    sample = dataset[CONFIG["sample_index"]]

    input_k = sample["input_kspace"].unsqueeze(0).to(device)
    target_k = sample["target_kspace"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)

    input_k_ri = to_ri(input_k)
    target_k_ri = to_ri(target_k)
    input_k_flat, mask_flat = flatten_adj_coils(input_k_ri, mask)

    center_idx = CONFIG["num_adj"] // 2
    input_k_center = input_k[:, center_idx, ...]
    input_k_center_ri = to_ri(input_k_center)
    zf_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(input_k_center_ri)), dim=1)

    with torch.no_grad():
        recon_img = model(input_k_flat, mask_flat, None)

    gt_img = rss_image_from_kspace(target_k_ri)

    zf_np = zf_img.cpu().numpy()[0]
    recon_np = recon_img.cpu().numpy()[0]
    gt_np = gt_img.cpu().numpy()[0]

    max_val = float(gt_np.max()) if float(gt_np.max()) > 0 else 1.0
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

    err_map = np.abs(gt_np - recon_np)

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
