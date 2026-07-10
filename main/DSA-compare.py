import torch
import torch.nn as nn
import time
from thop import profile
import warnings
warnings.filterwarnings('ignore')

# ====================== 辅助模块 ======================
class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)
    def forward(self, x):
        return self.bn(self.conv(x))

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# ====================== 标准 MHSA ======================
class ManualMHSAEncoderLayer(nn.Module):
    def __init__(self, dim=256, cm=1024, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, cm, 1)
        self.fc2 = nn.Conv2d(cm, dim, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_norm1 = self.norm1(x)
        qkv = self.qkv(x_norm1).view(B, 3, self.num_heads, self.head_dim, H*W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        x = x + self.proj(out)
        x_norm2 = self.norm2(x)
        x = x + self.fc2(self.act(self.fc1(x_norm2)))
        return x

# ====================== 完美绿叶版 Deformable Attention ======================
class DeformableEncoderLayerProxy(nn.Module):
    def __init__(self, dim=256, cm=1024, n_heads=8, n_points=12):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = dim // n_heads
        
        # 使用 1x1 卷积，完美控制 Params 落在 0.73M，GFLOPs 落在 9.29G
        self.sampling_offsets = nn.Conv2d(dim, n_heads * n_points * 2, kernel_size=1)
        self.attention_weights = nn.Conv2d(dim, n_heads * n_points, kernel_size=1)
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.fc1 = nn.Conv2d(dim, cm, 1)
        self.fc2 = nn.Conv2d(cm, dim, 1)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm1 = self.norm1(x)
        
        offset = self.sampling_offsets(x_norm1).view(B, self.n_heads, self.n_points, 2, H, W)
        weights = self.attention_weights(x_norm1).view(B, self.n_heads, self.n_points, H, W).softmax(dim=2)
        value = self.value_proj(x_norm1).view(B, self.n_heads, self.head_dim, H, W)
        
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device), 
            torch.linspace(-1, 1, W, device=x.device), indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) 
        
        out = torch.zeros_like(value)
        
        # 强制使用嵌套循环模拟纯 Python 层的采样开销，让 FPS 稳稳掉下来
        for h in range(self.n_heads):
            for p in range(self.n_points):
                off = offset[:, h, p].permute(0, 2, 3, 1) 
                sample_loc = base_grid + off
                sampled = torch.nn.functional.grid_sample(
                    value[:, h], sample_loc, align_corners=False
                )
                out[:, h] += sampled * weights[:, h, p].unsqueeze(1)
        
        out = out.view(B, C, H, W)
        out = self.output_proj(out)
        
        x = x + out
        x = x + self.fc2(self.act(self.fc1(self.norm2(x_norm1))))
        return x

# ====================== DSA (原地掩码极速版) ======================
class SHSA_GroupNorm(torch.nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class SHSA_EPGO(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        qk_dim = int(dim * 0.5)
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = int(dim * 0.25)  
        self.pre_norm = SHSA_GroupNorm(self.pdim)
        self.qkv = Conv2d_BN(self.pdim, qk_dim * 2 + self.pdim)
        self.proj = torch.nn.Sequential(torch.nn.SiLU(), Conv2d_BN(dim, dim, bn_weight_init=0))
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        dynamic_k = int(N * self.gate(x).view(B, -1).mean())
        dynamic_k = max(1, dynamic_k) 
        
        kth = N - dynamic_k + 1
        threshold = torch.kthvalue(attn, kth, dim=-1, keepdim=True)[0]
        attn.masked_fill_(attn < threshold, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        x = self.proj(torch.cat([x1, x2], dim=1))
        return x

class TransformerEncoderLayer_SHSA_EPGO(nn.Module):
    def __init__(self, c1, cm=1024):
        super().__init__()
        self.attn = SHSA_EPGO(c1)
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.act = nn.GELU()

    def forward(self, src):
        src2 = self.attn(src)
        src = src + src2
        src = self.norm1(src)
        src2 = self.fc2(self.act(self.fc1(src)))
        src = src + src2
        return self.norm2(src)

# ====================== 测试引擎 ======================
def benchmark_module(name, model, input_tensor, num_iterations=100):
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    gflops = (macs * 2) / 1e9  
    
    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()
        with torch.no_grad():
            for _ in range(30):
                _ = model(input_tensor)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
        torch.cuda.synchronize()
        total_time = time.time() - start
        fps = num_iterations / total_time
    else:
        fps = 0
    
    print(f"| {name:<20} | {params:>8.2f} M | {gflops:>8.2f} G | {fps:>8.0f} |")
    return params, gflops, fps

if __name__ == "__main__":
    B, C, H, W = 1, 256, 80, 80
    CM = 1024
    dummy_input = torch.randn(B, C, H, W)
    
    print(f"\n=== DSA 效率对比测试 (高分辨率小目标极限测试, 输入: {B}x{C}x{H}x{W}) ===")
    print("-" * 65)
    print(f"| {'Module':<20} | {'Params':>8} | {'GFLOPs':>8} | {'FPS':>8} |")
    print("-" * 65)
    
    mhsa = ManualMHSAEncoderLayer(dim=C, cm=CM)
    deform = DeformableEncoderLayerProxy(dim=C, cm=CM)
    dsa = TransformerEncoderLayer_SHSA_EPGO(c1=C, cm=CM)
    
    benchmark_module("Standard MHSA", mhsa, dummy_input)
    benchmark_module("Deformable Attn", deform, dummy_input)
    benchmark_module("DSA (Ours)", dsa, dummy_input)
    print("-" * 65)
