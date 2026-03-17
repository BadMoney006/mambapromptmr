import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import fastmri
from fastmri.losses import SSIMLoss
import random
import numpy as np

# 导入你的模块
from dataset import CineCMRDataset
from mamba_prompt_mr_model import MambaPromptMR

# ================= 配置区域 =================
CONFIG = {
    'root_path': r"/root/autodl-tmp/Cine/TrainingSet/AccFactor04", # 改成你的路径
    'batch_size': 1,           # 显存如果够大可以设为 2 或 4
    'num_epochs': 10,
    'lr': 1e-4,
    'num_adj': 5,              # 5帧输入
    'acc_factor': '04',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints',
    'log_interval': 10,
    'seed': 42,
    'min_lr': 2e-5,
    'frame_sampling': 'random_k',
    'num_sampled_frames': 3,
    'use_amp': True
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

def train():
    # 1. 准备工作
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    set_seed(CONFIG['seed'])
    device = torch.device(CONFIG['device'])
    print(f"🚀 使用设备: {device}")

    # 2. 数据加载
    print("📂 加载数据集...")
    train_dataset = CineCMRDataset(
        root_dir=CONFIG['root_path'],
        acc_factor=CONFIG['acc_factor'],
        num_adj=CONFIG['num_adj'],
        is_train=True,
        frame_sampling=CONFIG['frame_sampling'],
        num_sampled_frames=CONFIG['num_sampled_frames'],
        seed=CONFIG.get('seed', 42)
    )
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

    # 3. 模型初始化
    print("🧠 初始化 MambaPromptMR 模型...")
    # 注意: SingleCoil 数据通常是模拟的，Coil=1。但 MambaPromptMR 设计上支持多线圈。
    # Sensitivity Network 会自动处理 Coil 维度。
    model = MambaPromptMR(
        num_cascades=8,         # 级联数，显存不够改小 (e.g., 4)
        num_adj_slices=CONFIG['num_adj'],
        n_feat0=48,             # 特征维度
        img_size=256,           # 刚才 Padding 后的尺寸
        use_checkpoint=True     # 开启以节省显存
    ).to(device)

    # 4. 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=CONFIG['min_lr']
    )
    scaler = GradScaler(enabled=CONFIG['use_amp'])
    criterion_l1 = nn.L1Loss()
    criterion_ssim = SSIMLoss().to(device) # FastMRI 的 SSIM Loss

    # 5. 训练循环
    print("🔥 开始训练!")
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

        for i, batch in enumerate(progress_bar):
            # 搬运数据到 GPU
            # 输入: (B, T, C, H, W) -> 需要转为 FastMRI 格式 (B, T, C, H, W, 2) 如果是复数
            input_k = batch['input_kspace'].to(device)
            target_k = batch['target_kspace'].to(device) # (B, C, H, W) -> Label 是中心帧
            mask = batch['mask'].to(device)

            # 确保是复数 tensor，转为 (..., 2) 格式供 PromptMR 使用
            if input_k.is_complex():
                input_k = torch.view_as_real(input_k)
            if target_k.is_complex():
                target_k = torch.view_as_real(target_k)
            
            # mask 不需要最后一位，但也可能需要形状匹配
            # 现在的 Mask: (B, T, C, H, W) -> 正确

            # 前向传播
            # 模型输出: (B, T, H, W) 的模长图像 (Abs)
            # 前向传播
            # 模型输出: (B, T, H, W)
            with autocast(enabled=CONFIG['use_amp']):
                output_seq = model(input_k, mask)
            
            # 提取中心帧
            center_idx = CONFIG['num_adj'] // 2
            output_center = output_seq[:, center_idx, ...]
            
            # Target 图像
            target_img = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(target_k)), dim=1)
            
            # 计算 Loss
            with autocast(enabled=False):
                output_center_f = output_center.float()
                target_img_f = target_img.float()
                loss_l1 = criterion_l1(output_center_f, target_img_f)
                max_val = target_img_f.view(target_img_f.shape[0], -1).max(dim=1)[0]
                loss_ssim = criterion_ssim(output_center_f.unsqueeze(1), target_img_f.unsqueeze(1), data_range=max_val)
                loss = loss_l1 + 0.1 * loss_ssim
            
            # 反向传播
            optimizer.zero_grad()
            if CONFIG['use_amp']:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item(), 'L1': loss_l1.item()})

        # 保存模型
        print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_loss / len(train_loader):.6f}")
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], 'latest_model.pth'))
        if (epoch + 1) % 5 == 0:
             torch.save(model.state_dict(), os.path.join(CONFIG['save_dir'], f'epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    train()
