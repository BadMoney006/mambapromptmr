import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# 假设你已经有了 mamba_ssm 环境
# 从你提供的文件中导入必要的模块
# 如果在同一个文件，请确保相关类定义在上方
from mambaIR import BasicLayer, VSSBlock
from promptmr import PromptBlock, conv

class MambaLayerWrapper(nn.Module):
    """
    一个包装器，用于将 MambaIR 的 BasicLayer (处理 B, L, C) 
    适配到 U-Net 的 (B, C, H, W) 流程中。
    """
    def __init__(self, dim, depth, input_resolution, mlp_ratio=2., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution # (H, W)
        
        # 使用 mambaIR.py 中的 BasicLayer
        self.layer = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False # 根据显存情况开启
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H, W, C) -> (B, L, C)
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        # 传入 BasicLayer，需要提供 x_size=(H, W)
        # 注意：如果分辨率变化（因为 U-Net 的多尺度），
        # 这里的 input_resolution 可能需要动态调整或在 init 时固定
        # MambaIR 的实现通常依赖固定的 input_resolution 来做位置编码或初始化
        # 这里我们假设在不同尺度实例化不同的 Wrapper
        
        out = self.layer(x_flat, (H, W))
        
        # (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

class MambaDownBlock(nn.Module):
    def __init__(self, in_dim, out_dim, depth, input_resolution, kernel_size=3, bias=False):
        super().__init__()
        
        # 1. Mamba 特征提取 (替换原有的 Encoder CAB 堆叠)
        # 注意：输入分辨率在这一层是固定的
        self.encoder = MambaLayerWrapper(
            dim=in_dim, 
            depth=depth, 
            input_resolution=input_resolution
        )
        
        # 2. 下采样 (保持 PromptMR 的 Conv 下采样)
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=bias)

    def forward(self, x):
        enc = self.encoder(x)
        down = self.down(enc)
        return down, enc

class MambaUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, prompt_dim, depth, input_resolution, bias=False):
        super().__init__()
        
        # 计算融合后的通道数
        fuse_dim = in_dim + prompt_dim
        
        # 1. 融合 Prompt 的 Mamba 层 (替换原有的 fuse CAB)
        self.fuse_mamba = MambaLayerWrapper(
            dim=fuse_dim,
            depth=depth, 
            input_resolution=input_resolution # 此时还是低分辨率
        )
        
        # 2. 降维 (Reduce)
        self.reduce = nn.Conv2d(fuse_dim, in_dim, kernel_size=1, bias=bias)
        
        # 3. 上采样 (Upsample)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0, bias=False)
        )
        
        # 4. Refine Mamba 层 (替换原有的 ca CAB)
        # 上采样后的分辨率
        up_resolution = (input_resolution[0]*2, input_resolution[1]*2)
        self.refine_mamba = MambaLayerWrapper(
            dim=out_dim,
            depth=depth,
            input_resolution=up_resolution
        )

    def forward(self, x, prompt_dec, skip):
        # x: 来自深层 (低分辨率)
        # prompt_dec: 当前层的 Prompt
        # skip: 来自 Encoder 的跳跃连接 (高分辨率)
        
        # 1. 拼接 Prompt
        x = torch.cat([x, prompt_dec], dim=1)
        
        # 2. 融合特征 (Mamba)
        x = self.fuse_mamba(x)
        
        # 3. 降维
        x = self.reduce(x)
        
        # 4. 上采样并叠加 Skip
        x = self.up(x) + skip
        
        # 5. 精修特征 (Mamba)
        x = self.refine_mamba(x)
        
        return x

class MambaSkipBlock(nn.Module):
    def __init__(self, dim, depth, input_resolution):
        super().__init__()
        if depth == 0:
            self.skip_op = nn.Identity()
        else:
            self.skip_op = MambaLayerWrapper(dim, depth, input_resolution)

    def forward(self, x):
        return self.skip_op(x)

class MambaPromptUnet(nn.Module):
    def __init__(self, 
                 in_chans=10, 
                 out_chans=10, 
                 n_feat0=48,
                 img_size=256, # 必须指定输入图像大小，以便计算每层的分辨率
                 feature_dim = [72, 96, 120],
                 prompt_dim = [24, 48, 72],
                 len_prompt = [5, 5, 5],
                 prompt_size = [64, 32, 16],
                 # Mamba 深度配置 (替代原有的 n_enc_cab 等)
                 depths_enc = [2, 2, 2], 
                 depths_dec = [2, 2, 2],
                 depths_bottleneck = 2,
                 depths_skip = 1,
                 learnable_input_prompt=False,
                 bias=False,
                 ):
        super(MambaPromptUnet, self).__init__()

        # 0. 特征提取 (浅层特征)
        self.feat_extract = conv(in_chans, n_feat0, kernel_size=3, bias=bias)

        # 计算各级分辨率
        res_0 = (img_size, img_size)
        res_1 = (img_size // 2, img_size // 2)
        res_2 = (img_size // 4, img_size // 4)
        res_3 = (img_size // 8, img_size // 8)

        # 1. Encoder (基于 Mamba)
        # Level 1: n_feat0 -> feature_dim[0]
        self.enc_level1 = MambaDownBlock(n_feat0, feature_dim[0], depths_enc[0], res_0, bias=bias)
        # Level 2: feature_dim[0] -> feature_dim[1]
        self.enc_level2 = MambaDownBlock(feature_dim[0], feature_dim[1], depths_enc[1], res_1, bias=bias)
        # Level 3: feature_dim[1] -> feature_dim[2]
        self.enc_level3 = MambaDownBlock(feature_dim[1], feature_dim[2], depths_enc[2], res_2, bias=bias)

        # 2. Skip Connections
        self.skip_attn1 = MambaSkipBlock(n_feat0, depths_skip, res_0)
        self.skip_attn2 = MambaSkipBlock(feature_dim[0], depths_skip, res_1)
        self.skip_attn3 = MambaSkipBlock(feature_dim[1], depths_skip, res_2)

        # 3. Bottleneck (基于 Mamba)
        self.bottleneck = MambaLayerWrapper(feature_dim[2], depths_bottleneck, res_3)

        # 4. Decoder (基于 Mamba + Prompt)
        # Level 3: feature_dim[2] -> feature_dim[1]
        self.prompt_level3 = PromptBlock(prompt_dim=prompt_dim[2], prompt_len=len_prompt[2], prompt_size=prompt_size[2], lin_dim=feature_dim[2], learnable_input_prompt=learnable_input_prompt)
        self.dec_level3 = MambaUpBlock(feature_dim[2], feature_dim[1], prompt_dim[2], depths_dec[2], input_resolution=res_3, bias=bias)

        # Level 2: feature_dim[1] -> feature_dim[0]
        self.prompt_level2 = PromptBlock(prompt_dim=prompt_dim[1], prompt_len=len_prompt[1], prompt_size=prompt_size[1], lin_dim=feature_dim[1], learnable_input_prompt=learnable_input_prompt)
        self.dec_level2 = MambaUpBlock(feature_dim[1], feature_dim[0], prompt_dim[1], depths_dec[1], input_resolution=res_2, bias=bias)

        # Level 1: feature_dim[0] -> n_feat0
        self.prompt_level1 = PromptBlock(prompt_dim=prompt_dim[0], prompt_len=len_prompt[0], prompt_size=prompt_size[0], lin_dim=feature_dim[0], learnable_input_prompt=learnable_input_prompt)
        self.dec_level1 = MambaUpBlock(feature_dim[0], n_feat0, prompt_dim[0], depths_dec[0], input_resolution=res_1, bias=bias)

        # 5. 输出层
        self.conv_last = conv(n_feat0, out_chans, 5, bias=bias)

    def forward(self, x):
        # x shape: (B, C, H, W)
        
        # 0. 浅层特征
        x = self.feat_extract(x)

        # 1. Encoder
        # x_enc1 是输入到下层之前的特征（用于Skip），enc1 是下采样后的结果
        # MambaDownBlock 返回 (downsampled, feature_before_down)
        x_down1, enc1 = self.enc_level1(x)
        x_down2, enc2 = self.enc_level2(x_down1)
        x_down3, enc3 = self.enc_level3(x_down2)

        # 2. Bottleneck
        x_bot = self.bottleneck(x_down3)

        # 3. Decoder
        # Prompt 生成依赖于 bottleneck 或 上一层的输出
        dec_prompt3 = self.prompt_level3(x_bot)
        # Decoder 输入：(深层特征, Prompt, Skip特征)
        x_up3 = self.dec_level3(x_bot, dec_prompt3, self.skip_attn3(enc3))

        dec_prompt2 = self.prompt_level2(x_up3)
        x_up2 = self.dec_level2(x_up3, dec_prompt2, self.skip_attn2(enc2))

        dec_prompt1 = self.prompt_level1(x_up2)
        x_up1 = self.dec_level1(x_up2, dec_prompt1, self.skip_attn1(enc1))

        # 4. 输出
        return self.conv_last(x_up1)

# --------------------------------------------------------
# 使用示例
# --------------------------------------------------------
if __name__ == "__main__":
    # 假设输入尺寸为 256x256, 10通道 (2通道复数 * 5帧)
    model = MambaPromptUnet(
        in_chans=10,
        out_chans=10,
        img_size=256,
        n_feat0=48
    ).cuda()
    
    input_tensor = torch.randn(1, 10, 256, 256).cuda()
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    
    # 打印参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params / 1e6:.2f} M")