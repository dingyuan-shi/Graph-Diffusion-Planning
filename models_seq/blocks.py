import torch
import torch.nn as nn
import numpy as np
import einops
from einops.layers.torch import Rearrange
from einops import rearrange

    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        # 2i * -\log 10000^{1/(d-1)}
        # -(2i/d) * log(10000)
        self.emb = torch.exp(-torch.arange(0, self.dim, 2) / dim * np.log(10000)).to(self.device)
        
    def forward(self, x):
        # x: batch of times
        emb = x.view(-1, 1) * self.emb.unsqueeze(0)
        encodings = torch.zeros(x.shape[0], self.dim, device=self.device)
        encodings[:, 0::2] = emb.sin()
        encodings[:, 1::2] = emb.cos()
        return encodings



class Conv1dBlock(nn.Module):

    def __init__(self, i_channels, o_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            Rearrange("b h c -> b c h"), 
            nn.Conv1d(i_channels, o_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('b c h -> b h c'), 
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LinearAttention(nn.Module):
    def __init__(self, dim, device, heads=4, dim_head=32):
        super().__init__()
        self.device = device
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False, device=device)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1, device=device)

    def forward(self, x, lengths):
        if lengths is not None:
            b, h, c = x.shape
            mask = torch.zeros(b, h + 1, c).long().to(self.device)
            mask[torch.arange(mask.shape[0]), lengths] = 1
            mask = mask.cumsum(dim=1)
            mask = mask[:, :h, :]
            x = torch.masked_fill(x, mask==1, 0)

        x = rearrange(x, "b d c -> b c d")
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) d -> b h c d", h=self.heads), qkv)
        q = q * self.scale
        if lengths is not None:
            b, h, c, d = q.shape
            mask = torch.zeros(b, h, c, d + 1).long().to(self.device)
            mask[torch.arange(lengths.shape[0]), :, :, lengths] = 1
            mask = mask.cumsum(dim=-1)[:, :, :, :d]
            # mask q and k
            q = torch.masked_fill(q, mask==1, 0)
            k = torch.masked_fill(k, mask==1, -1e15)

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c d -> b (h c) d')
        out = self.to_out(out)
        return rearrange(out, "b c d -> b d c")
