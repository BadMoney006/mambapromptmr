import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F


class CineCMRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        acc_factor='04',
        num_adj=5,
        is_train=True,
        transform=None,
        frame_sampling="all",
        num_sampled_frames=3,
        seed=42,
    ):
        """
        CMRxRecon2023 Cine 数据集加载器 (最终修复版)
        """
        self.root_dir = Path(root_dir)
        self.acc_factor = acc_factor
        self.num_adj = num_adj
        self.is_train = is_train
        self.transform = transform
        self.frame_sampling = frame_sampling
        self.num_sampled_frames = num_sampled_frames
        self.rng = np.random.RandomState(seed)
        
        # 自动推断 FullSample 目录
        self.full_dir = self.root_dir.parent / 'FullSample'
        if not self.full_dir.exists() and is_train:
            print(f"警告: 未找到 FullSample 目录: {self.full_dir}。将仅返回输入数据。")
            self.full_dir = None

        self.samples = []
        self._mask_cache = {}
        self._warned_missing_mask = set()
        self._scan_dataset()

    def _scan_dataset(self):
        print(f"正在扫描数据集: {self.root_dir} ...")
        subject_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('P')])
        
        for subj_dir in subject_dirs:
            for view in ['sax', 'lax']:
                fname = f"cine_{view}.mat"
                file_path = subj_dir / fname
                
                full_path = None
                if self.full_dir:
                    full_path = self.full_dir / subj_dir.name / fname
                    if not full_path.exists():
                        continue
                else:
                    full_path = file_path 

                if file_path.exists():
                    try:
                        with h5py.File(file_path, 'r') as f:
                            keys = list(f.keys())
                            target_key = None
                            if f'kspace_sub{self.acc_factor}' in keys:
                                target_key = f'kspace_sub{self.acc_factor}'
                            elif f'kspace_single_sub{self.acc_factor}' in keys:
                                target_key = f'kspace_single_sub{self.acc_factor}'
                            
                            if target_key:
                                # --- 修复核心：兼容 Group 和 Dataset ---
                                obj = f[target_key]
                                shape = None
                                try:
                                    shape = obj['real'].shape # 尝试读取结构体
                                except:
                                    shape = obj.shape # 直接读取
                                # -----------------------------------
                                
                                num_t = shape[0]
                                num_slices = shape[1]
                                frame_indices = self._get_frame_indices(num_t)
                                for s in range(num_slices):
                                    for t in frame_indices:
                                        self.samples.append({
                                            'sub_path': str(file_path),
                                            'full_path': str(full_path),
                                            'slice_idx': s,
                                            'frame_idx': t,
                                            'num_frames': num_t,
                                            'subject': subj_dir.name,
                                            'view': view
                                        })
                    except Exception as e:
                        print(f"读取错误 {file_path}: {e}")
        print(f"数据集扫描完成: 共找到 {len(self.samples)} 个样本。")

    def _read_kspace(self, file_path, key_suffix, slice_idx):
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            is_single = 'kspace_single_full' in keys or f'kspace_single_{key_suffix}' in keys
            prefix = 'kspace_single_' if is_single else 'kspace_'
            target_key = f"{prefix}{key_suffix}"
            
            data_ptr = f[target_key]
            try:
                real = data_ptr['real'][:]
                imag = data_ptr['imag'][:]
                kspace = real + 1j * imag
            except:
                kspace = data_ptr[:]

            kspace = kspace[:, slice_idx, ...]
            if kspace.ndim == 4:
                kspace = kspace.transpose(0, 1, 3, 2)
            elif kspace.ndim == 3:
                kspace = kspace.transpose(0, 2, 1)
            return kspace

    def _get_frame_indices(self, num_t):
        mode = (self.frame_sampling or "all").lower()
        if mode == "all":
            return list(range(num_t))
        if mode == "center":
            return [num_t // 2]
        if mode == "random_one":
            return [-1]
        if mode == "random_k":
            k = max(1, int(self.num_sampled_frames))
            k = min(k, num_t)
            return sorted(self.rng.choice(num_t, size=k, replace=False).tolist())
        return list(range(num_t))

    def _read_mask(self, file_path):
        if file_path in self._mask_cache:
            return self._mask_cache[file_path]
        mask_path = file_path.replace(".mat", "_mask.mat")
        if not os.path.exists(mask_path):
            if file_path not in self._warned_missing_mask:
                print(f"⚠️ 未找到 mask 文件，回退使用非零 mask: {mask_path}")
                self._warned_missing_mask.add(file_path)
            self._mask_cache[file_path] = None
            return None
        with h5py.File(mask_path, "r") as f:
            key = list(f.keys())[0]
            mask = f[key][()]
        if mask.ndim >= 2:
            mask = mask[:, 0]
        mask = np.asarray(mask).squeeze()
        self._mask_cache[file_path] = mask
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        kspace_sub = self._read_kspace(sample['sub_path'], f"sub{self.acc_factor}", sample['slice_idx'])
        kspace_full = self._read_kspace(sample['full_path'], "full", sample['slice_idx'])
        
        num_frames = kspace_sub.shape[0]
        t = sample.get('frame_idx', num_frames // 2)
        if t is None or int(t) < 0:
            t = int(self.rng.randint(0, num_frames))

        center_idx = self.num_adj // 2
        adj_indices = [ (t + offset) % num_frames for offset in range(-center_idx, center_idx + 1) ]
            
        kspace_sub_adj = kspace_sub[adj_indices]
        kspace_full_center = kspace_full[t]
        
        kspace_sub_tensor = torch.from_numpy(kspace_sub_adj).cfloat()
        kspace_full_tensor = torch.from_numpy(kspace_full_center).cfloat()

        # Ensure coil dimension exists before building mask
        if kspace_sub_tensor.dim() == 3:
            kspace_sub_tensor = kspace_sub_tensor.unsqueeze(1)
            kspace_full_tensor = kspace_full_tensor.unsqueeze(0)
        
        # mask = torch.abs(kspace_sub_tensor).sum(dim=-1).sum(dim=-1) > 0
        # [修改后] 这是空间 Mask，正确反映采样位置
        # 只要该像素点幅值大于0，就认为是被采样的
        mask_1d = self._read_mask(sample['sub_path'])
        if mask_1d is not None:
            num_cols = kspace_sub_tensor.shape[-2]
            mask_1d = mask_1d.astype(np.float32)
            if mask_1d.shape[0] != num_cols:
                mask_1d = np.resize(mask_1d, (num_cols,))
            mask_col = torch.from_numpy(mask_1d).view(1, 1, num_cols, 1)
            mask = mask_col.expand(kspace_sub_tensor.shape[0], kspace_sub_tensor.shape[1], num_cols, kspace_sub_tensor.shape[-1])
        else:
            mask = (torch.abs(kspace_sub_tensor) > 0).float()
        mask = mask.float()
        
        # 单线圈增加维度: (T, H, W) -> (T, 1, H, W)
        # --- 新增：Padding 到 16 的倍数 (防止 U-Net 尺寸不匹配报错) ---
        # 目标: H=246 -> 256
        pad_h = (16 - kspace_sub_tensor.shape[-2] % 16) % 16
        pad_w = (16 - kspace_sub_tensor.shape[-1] % 16) % 16
        
        if pad_h > 0 or pad_w > 0:
            # F.pad 参数顺序: (Left, Right, Top, Bottom)
            # 记得对 kspace_sub, kspace_full, mask 都做 padding
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            kspace_sub_tensor = F.pad(kspace_sub_tensor, padding, mode='constant', value=0)
            kspace_full_tensor = F.pad(kspace_full_tensor, padding, mode='constant', value=0)
            mask = F.pad(mask, padding, mode='constant', value=0)
        # -----------------------------------------------------------

        return {
            'input_kspace': kspace_sub_tensor, # (5, Coil, H, W)
            'target_kspace': kspace_full_tensor, # (Coil, H, W)
            'mask': mask,
            'metadata': {
                'subject': sample['subject'],
                'slice': sample['slice_idx'],
                'frame_idx': t
            }
        }
