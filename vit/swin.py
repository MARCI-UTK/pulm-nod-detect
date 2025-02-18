import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, Heads, N, C/Heads)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        
    def forward(self, x):
        B, D, H, W, C = x.shape
        x = x.view(B, D//2, 2, H//2, 2, W//2, 2, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B, (D//2) * (H//2) * (W//2), 8 * C)
        return self.reduction(x)

class SwinTransformer3D(nn.Module):
    def __init__(self, input_dim=1, embed_dim=96, depths=[2, 2, 6], num_heads=[3, 6, 12], window_size=(4,4,4), num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv3d(input_dim, embed_dim, kernel_size=4, stride=4)
        
        layers = []
        for i in range(len(depths)):
            for _ in range(depths[i]):
                layers.append(SwinTransformerBlock3D(embed_dim, num_heads[i], window_size))
            if i < len(depths) - 1:
                layers.append(PatchMerging3D(embed_dim))
                embed_dim *= 2
        
        self.layers = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)  # (B, C, D, H, W) -> (B, C', D/4, H/4, W/4)
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        x = self.layers(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)

# Example usage
if __name__ == "__main__":
    model = SwinTransformer3D(input_dim=1, num_classes=10)
    input_tensor = torch.randn(1, 1, 32, 32, 32)  # (Batch, Channels, Depth, Height, Width)
    output = model(input_tensor)
    print(output.shape)  # Expected: (1, 10)