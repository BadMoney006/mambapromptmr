import torch
import torch.nn as nn
import fastmri
from typing import List, Tuple, Optional
from promptmr import PromptMRBlock, SensitivityModel
from mamba_prompt_unet import MambaPromptUnet 

# NormMambaPromptUnet 类保持不变...
class NormMambaPromptUnet(nn.Module):
    def __init__(self, num_adj_slices=5, n_feat0=48, img_size=256):
        super().__init__()
        in_chans = num_adj_slices * 2
        self.net = MambaPromptUnet(
            in_chans=in_chans, out_chans=in_chans, n_feat0=n_feat0, img_size=img_size
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)
        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x = self.net(x)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)
        return x

class MambaPromptMR(nn.Module):
    def __init__(self,
                 num_cascades: int = 8,
                 num_adj_slices: int = 5,
                 img_size: int = 256,
                 n_feat0: int = 48,
                 use_checkpoint: bool = True):
        super().__init__()
        
        self.sens_net = SensitivityModel(
            num_adj_slices=num_adj_slices,
            n_feat0=24,
            mask_center=True
        )
        
        self.cascades = nn.ModuleList([
            PromptMRBlock(
                model=NormMambaPromptUnet(
                    num_adj_slices=num_adj_slices,
                    n_feat0=n_feat0,
                    img_size=img_size
                ),
                num_adj_slices=num_adj_slices
            )
            for _ in range(num_cascades)
        ])
        
        self.use_checkpoint = use_checkpoint

    def forward(self, masked_kspace, mask):
        """
        Args:
            masked_kspace: (B, Time, Coil, H, W, 2)
            mask: (B, Time, Coil, H, W)
        """
        # --- 修复 1：维度重塑 (Flatten Time & Coil) ---
        if masked_kspace.dim() == 6:
            B, T, C, H, W, two = masked_kspace.shape
            masked_kspace = masked_kspace.view(B, T * C, H, W, 2)
            
            # Mask 同步 reshape: (B, T, C, H, W) -> (B, T*C, H, W)
            if mask.dim() == 5:
                mask = mask.view(B, T * C, H, W)
        
        # --- 修复 2：增加 Mask 最后一个维度 (B, ..., 1) ---
        if mask.dim() == 4:
            mask = mask.unsqueeze(-1) # 变成 (B, TC, H, W, 1)

        # --- 修复 3：类型强制转换 (Float -> Bool) ---
        # 解决 "where expected condition to be a boolean tensor"
        if not mask.dtype == torch.bool:
            mask = mask > 0 # 强转为 Boolean
        # ---------------------------------------------

        # 1. 估计灵敏度图
        if self.use_checkpoint and self.training:
            sens_maps = torch.utils.checkpoint.checkpoint(
                self.sens_net, masked_kspace, mask, None, use_reentrant=False)
        else:
            sens_maps = self.sens_net(masked_kspace, mask, None)
            
        # 2. 迭代重建
        kspace_pred = masked_kspace.clone()
        
        for cascade in self.cascades:
            if self.use_checkpoint and self.training:
                kspace_pred = torch.utils.checkpoint.checkpoint(
                    cascade, kspace_pred, masked_kspace, mask, sens_maps, use_reentrant=False)
            else:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
                
        # 3. 最终输出图像
        final_image = self.cascades[0].sens_reduce(kspace_pred, sens_maps)
        
        return fastmri.complex_abs(final_image)