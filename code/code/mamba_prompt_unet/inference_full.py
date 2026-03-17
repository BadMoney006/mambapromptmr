import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import fastmri
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import os
import matplotlib.pyplot as plt

# 导入你的模型
from mamba_prompt_mr_model import MambaPromptMR

# ================= 配置区域 =================
CONFIG = {
    # 你的测试集路径
    'val_root': r"/root/autodl-tmp/TestSet/AccFactor04", 
    'model_path': '/root/autodl-tmp/mamba_prompt_unet/checkpoints/latest_model.pth',
    'acc_factor': '04',
    'num_adj': 5,
    'frame_sampling': 'all',
    'num_sampled_frames': 3,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_results': True,
    'save_dir': './eval_results',
    'use_crop_eval': True
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
    # Align with cmrxrecon crop_submission spatial ratio: H/3, W/2
    crop_h = matlab_round(x.shape[-2] / 3)
    crop_w = matlab_round(x.shape[-1] / 2)
    return center_crop_hw(x, crop_h, crop_w)

def get_frame_indices(num_t, mode, num_sampled_frames=3, seed=42):
    mode = (mode or "all").lower()
    if mode == "all":
        return list(range(num_t))
    if mode == "center":
        return [num_t // 2]
    rng = np.random.RandomState(seed)
    if mode == "random_one":
        return [int(rng.randint(0, num_t))]
    if mode == "random_k":
        k = max(1, int(num_sampled_frames))
        k = min(k, num_t)
        return sorted(rng.choice(num_t, size=k, replace=False).tolist())
    return list(range(num_t))

class CineEvalDataset(Dataset):
    def __init__(self, root_dir, acc_factor='04'):
        self.root_dir = Path(root_dir)
        self.acc_factor = acc_factor
        self.samples = []
        self.has_gt = False # 标记是否找到 GT
        
        # 尝试寻找 FullSample
        # 逻辑：如果 val_root 是 .../TestSet/AccFactor04
        # parent 是 .../TestSet，下面应该有 FullSample
        self.full_dir = self.root_dir.parent / 'FullSample'
        if self.full_dir.exists():
            self.has_gt = True
            print(f"✅ 发现 Ground Truth 目录: {self.full_dir}")
        else:
            print(f"⚠️ 未找到 GT 目录: {self.full_dir}。将仅执行推理，不计算 PSNR。")

        self._scan_dataset()

    def _scan_dataset(self):
        print(f"正在扫描路径: {self.root_dir} ...")
        subject_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('P')])
        
        for subj_dir in subject_dirs:
            for view in ['sax', 'lax']:
                fname = f"cine_{view}.mat"
                file_path = subj_dir / fname
                
                # 只有当 GT 目录存在且文件存在时，才记录 full_path
                full_path = None
                if self.has_gt:
                    potential_gt = self.full_dir / subj_dir.name / fname
                    if potential_gt.exists():
                        full_path = potential_gt

                if file_path.exists():
                    try:
                        with h5py.File(file_path, 'r') as f:
                            keys = list(f.keys())
                            # 自动匹配 Key
                            target_key = None
                            if f'kspace_sub{self.acc_factor}' in keys: target_key = f'kspace_sub{self.acc_factor}'
                            elif f'kspace_single_sub{self.acc_factor}' in keys: target_key = f'kspace_single_sub{self.acc_factor}'
                            
                            if target_key:
                                obj = f[target_key]
                                shape = None
                                # --- 修复扫描逻辑 ---
                                try:
                                    shape = obj['real'].shape # 尝试读取 struct
                                except:
                                    shape = obj.shape # 尝试直接读取 dataset
                                # -------------------
                                
                                num_slices = shape[1] 
                                for s in range(num_slices):
                                    self.samples.append({
                                        'sub_path': str(file_path),
                                        'full_path': str(full_path) if full_path else None,
                                        'slice_idx': s,
                                        'subject': subj_dir.name,
                                        'view': view
                                    })
                    except Exception as e:
                        # print(f"Skip {file_path}: {e}")
                        pass
        print(f"✅ 扫描完成: 共 {len(self.samples)} 个切片。")

    def _read_kspace_slice(self, file_path, key_prefix, slice_idx):
        if file_path is None: return None
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            actual_key = None
            for k in keys:
                if key_prefix in k:
                    actual_key = k
                    break
            if actual_key is None and 'kspace_full' in keys: actual_key = 'kspace_full'
            
            if actual_key is None: return None

            data_ptr = f[actual_key]
            try:
                real = data_ptr['real'][:]
                imag = data_ptr['imag'][:]
                kspace = real + 1j * imag
            except:
                kspace = data_ptr[:]
            return kspace[:, slice_idx, ...]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取输入
        kspace_sub = self._read_kspace_slice(sample['sub_path'], f"sub{self.acc_factor}", sample['slice_idx'])
        
        # 读取 GT (如果有)
        kspace_full = None
        if sample['full_path']:
            kspace_full = self._read_kspace_slice(sample['full_path'], "full", sample['slice_idx'])
        
        kspace_sub = torch.from_numpy(kspace_sub).cfloat()
        if kspace_sub.dim() == 3: kspace_sub = kspace_sub.unsqueeze(1)
        
        if kspace_full is not None:
            kspace_full = torch.from_numpy(kspace_full).cfloat()
            if kspace_full.dim() == 3: kspace_full = kspace_full.unsqueeze(1)
        else:
            # 如果没有 GT，造一个假的占位符，防止 DataLoader 报错
            kspace_full = torch.zeros_like(kspace_sub)

        mask = (torch.abs(kspace_sub) > 0).float()

        return {
            'input_kspace': kspace_sub,
            'target_kspace': kspace_full,
            'mask': mask,
            'has_gt': kspace_full is not None and sample['full_path'] is not None,
            'meta': sample
        }

def evaluate():
    device = torch.device(CONFIG['device'])
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    val_dataset = CineEvalDataset(root_dir=CONFIG['val_root'], acc_factor=CONFIG['acc_factor'])
    if len(val_dataset) == 0:
        print("❌ 错误: 依然没有找到切片，请检查路径是否包含子文件夹结构。")
        return
        
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"🧠 加载模型: {CONFIG['model_path']}")
    # ⚠️ 请确保这里参数与 train.py 一致
    model = MambaPromptMR(
        num_cascades=8,         
        num_adj_slices=CONFIG['num_adj'],
        n_feat0=48,             
        img_size=256,
        use_checkpoint=False
    ).to(device)
    
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    total_psnr = []
    total_ssim = []

    print("🚀 开始推理...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            input_vol = batch['input_kspace'].to(device)
            target_vol = batch['target_kspace'].to(device)
            mask_vol = batch['mask'].to(device)
            has_gt = batch['has_gt'].item() # bool
            
            if input_vol.is_complex(): input_vol = torch.view_as_real(input_vol)
            if target_vol.is_complex(): target_vol = torch.view_as_real(target_vol)

            B, T, C, H, W, _ = input_vol.shape
            
            # --- Padding ---
            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16
            if pad_h > 0 or pad_w > 0:
                input_vol = input_vol.permute(0, 1, 2, 5, 3, 4)
                input_vol = F.pad(input_vol, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
                input_vol = input_vol.permute(0, 1, 2, 4, 5, 3)
                
                mask_vol = mask_vol.permute(0, 1, 2, 3, 4)
                mask_vol = F.pad(mask_vol, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
                
                target_vol = target_vol.permute(0, 1, 2, 5, 3, 4)
                target_vol = F.pad(target_vol, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
                target_vol = target_vol.permute(0, 1, 2, 4, 5, 3)
            
            # --- 滑动窗口推理 ---
            frame_indices = get_frame_indices(
                T,
                CONFIG['frame_sampling'],
                CONFIG['num_sampled_frames'],
                CONFIG['seed'],
            )
            recon_slice = []
            for t in frame_indices:
                idx_list = []
                center_offset = CONFIG['num_adj'] // 2
                for offset in range(-center_offset, center_offset + 1):
                    curr_t = max(0, min(t + offset, T - 1))
                    idx_list.append(curr_t)
                
                input_window = input_vol[:, idx_list, ...] 
                mask_window = mask_vol[:, idx_list, ...]
                
                output_window = model(input_window, mask_window)
                pred_t = output_window[:, center_offset, ...]
                
                # Unpadding
                if pad_h > 0 or pad_w > 0:
                    h_start = pad_h // 2
                    w_start = pad_w // 2
                    pred_t = pred_t[..., h_start:h_start+H, w_start:w_start+W]
                
                recon_slice.append((t, pred_t))
            
            recon_slice = sorted(recon_slice, key=lambda x: x[0])
            recon_indices = [t for t, _ in recon_slice]
            recon_vol = torch.cat([p for _, p in recon_slice], dim=0) # (Ts, H, W)
            recon_np = recon_vol.cpu().numpy()

            # --- 如果有 GT，计算指标 ---
            if has_gt:
                target_vol_flat = target_vol.squeeze(0)
                gt_vol = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(target_vol_flat)), dim=1)
                gt_np = gt_vol.cpu().numpy()
                
                if pad_h > 0 or pad_w > 0:
                    h_start = pad_h // 2
                    w_start = pad_w // 2
                    gt_np = gt_np[..., h_start:h_start+H, w_start:w_start+W]
                
                max_val = gt_np.max()
                slice_psnr = []
                slice_ssim = []
                for i, t in enumerate(recon_indices):
                    if CONFIG['use_crop_eval']:
                        gt_k = crop_for_cmrx_cine(gt_np[t])
                        recon_k = crop_for_cmrx_cine(recon_np[i])
                    else:
                        gt_k = gt_np[t]
                        recon_k = recon_np[i]
                    p = psnr_metric(gt_k, recon_k, data_range=max_val)
                    s = ssim_metric(gt_k, recon_k, data_range=max_val)
                    slice_psnr.append(p)
                    slice_ssim.append(s)
                
                avg_psnr = np.mean(slice_psnr)
                avg_ssim = np.mean(slice_ssim)
                total_psnr.append(avg_psnr)
                total_ssim.append(avg_ssim)

            # --- 保存可视化 (前3个样本) ---
            if i < 3:
                subj_name = batch['meta']['subject'][0]
                slice_idx = batch['meta']['slice_idx'][0]
                
                plt.figure(figsize=(10, 5))
                # 显示中间一帧
                mid_i = len(recon_indices) // 2 if recon_indices else 0
                plt.subplot(1, 2, 1)
                plt.imshow(recon_np[mid_i], cmap='gray')
                plt.title(f"Recon {subj_name} Slice{slice_idx}")
                plt.axis('off')
                
                if has_gt:
                    plt.subplot(1, 2, 2)
                    if recon_indices:
                        plt.imshow(gt_np[recon_indices[mid_i]], cmap='gray')
                    else:
                        plt.imshow(gt_np[0], cmap='gray')
                    plt.title("Ground Truth")
                    plt.axis('off')
                
                plt.savefig(f"{CONFIG['save_dir']}/{subj_name}_slice{slice_idx}.png")
                plt.close()

    if total_psnr:
        print("\n" + "="*40)
        print(f"📊 最终测试报告")
        print(f"平均 PSNR: {np.mean(total_psnr):.4f} dB")
        print(f"平均 SSIM: {np.mean(total_ssim):.4f}")
        print("="*40)
    else:
        print("\n⚠️ 未计算 PSNR/SSIM (因为未找到 Ground Truth)")
        print(f"所有重建结果已生成完毕。")

if __name__ == "__main__":
    evaluate()