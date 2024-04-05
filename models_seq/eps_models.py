import torch 
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange
from einops.layers.torch import Rearrange
import pickle

from models_seq.blocks import (
    SinusoidalPosEmb, 
    Conv1dBlock, 
    Residual, 
    LinearAttention, 
)


class XTResBlock(nn.Module):

    def __init__(self, x_in_dim, t_in_dim, out_dim, device, kernel_size=5):
        super().__init__()
        self.device = device
        self.block1 = Conv1dBlock(x_in_dim, out_dim, kernel_size).to(device)
        self.block2 = Conv1dBlock(out_dim, out_dim, kernel_size).to(device)

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(t_in_dim, out_dim, device=device),
            Rearrange("b c -> b 1 c") 
        )
        self.residual_conv = nn.Sequential(
            Rearrange("b h c -> b c h"), 
            nn.Conv1d(x_in_dim, out_dim, 1, device=device) if x_in_dim != out_dim else nn.Identity(), 
            Rearrange("b c h -> b h c")
        )

    def forward(self, x, t):
        '''
            x : b h c
            t : b h d
            returns:
            out : b h e
        '''
        out = self.block1(x) + self.time_mlp(t)
        out = self.block2(out)
        return out + self.residual_conv(x)
    
    
class UnetBlock(nn.Module):
    def __init__(self, x_dim, time_dim, out_dim, device, down_up, last):
        super().__init__()
        self.device = device
        self.xtblock1 = XTResBlock(x_dim, time_dim, out_dim, device, kernel_size=5)
        self.xtblock2 = XTResBlock(out_dim, time_dim, out_dim, device, kernel_size=5)
        self.attn = Residual(LinearAttention(out_dim, device))
        self.down = down_up == "down"
        self.sample = nn.Identity()
        
    def forward(self, xs, lengths, ts):
        x = self.xtblock1(xs, ts)
        x = self.xtblock2(x, ts)
        h = self.attn(x, lengths)
        x = self.sample(h)
        return x, h


class EPSM(nn.Module):
    
    def __init__(self, n_vertex, x_emb_dim, dims, hidden_dim, device, pretrain_path=None):
        super().__init__()
        time_dim = hidden_dim
        self.device = device
        # temporal embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim, device), 
            nn.Linear(time_dim, 4 * time_dim, device=device), 
            nn.Mish(), 
            nn.Linear(4 * time_dim, time_dim, device=device)
        )
        # n_vertex denotes <end>,  n_vertex + 1 denotes <padding>
        if pretrain_path is not None:
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert n_vertex == len(node2vec)
            if x_emb_dim != node2vec[0].shape[0]:
                print("Use pretrained embed dims")
            x_emb_dim = node2vec[0].shape[0]
            nodeemb = torch.zeros(n_vertex + 2, x_emb_dim)
            for k in node2vec:
                nodeemb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(nodeemb, freeze=False).to(device)
        else:
            self.x_embedding = nn.Embedding(n_vertex + 2, x_emb_dim, padding_idx=n_vertex, device=device)
        
        in_out_dim = [(a, b) for a, b in zip(dims, dims[1:])]
        print(in_out_dim)
        # down blocks
        self.down_blocks = []
        n_reso = len(in_out_dim)
        for k, (in_dim, out_dim) in enumerate(in_out_dim):
            self.down_blocks.append(UnetBlock(
                in_dim, time_dim, out_dim, device, 
                down_up="down", last=(k == n_reso - 1)))
        
        # middle parts
        mid_dim = dims[-1]
        self.mid_block1 = XTResBlock(mid_dim, time_dim, mid_dim, device)
        self.mid_attn = Residual(LinearAttention(mid_dim, device))
        self.mid_block2 = XTResBlock(mid_dim, time_dim, mid_dim, device)

        # up blocks
        self.up_blocks = []
        for k, (out_dim, in_dim) in enumerate(reversed(in_out_dim[1:])):
            self.up_blocks.append(UnetBlock(
                in_dim * 2, time_dim, out_dim, device, 
                down_up="up", last=(k == n_reso - 1)))
        
        # final parts
        self.final_conv = nn.Sequential(
            Conv1dBlock(in_out_dim[1][0], dims[0], kernel_size=5),
            Rearrange("b h c -> b c h"), 
            nn.Conv1d(dims[0], n_vertex, 1, device=device),
            Rearrange("b c h -> b h c")
        ).to(device)
        
        
    def forward(self, xt_padded, lengths, t):
        # xt_padded: shape b, h, each is a xt label
        # t: shape b
        t = self.time_mlp(t)
        x = self.x_embedding(xt_padded)
        hiddens = []
        for k, down_block in enumerate(self.down_blocks):
            x, h = down_block(x, lengths if k == 0 else None, t)
            hiddens.append(h)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, None)
        x = self.mid_block2(x, t)
        
        for up_block in self.up_blocks:
            x = torch.cat((x, hiddens.pop()), dim=-1)
            x, _ = up_block(x, None, t)
        x = self.final_conv(x)
        return x