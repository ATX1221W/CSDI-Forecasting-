
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, heads, max_seq_len, n_local_attn_heads=0, local_attn_window_size=0):
        super().__init__()
        self.layers = nn.ModuleList([LinearAttentionLayer(dim, heads) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LinearAttentionLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, n, d = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, -1).transpose(1, 2), qkv)
        q = F.softmax(q, dim=-1)
        k = F.softmax(k, dim=-2)
        context = torch.einsum('bnd,bne->bde', k, v)
        out = torch.einsum('bnd,bde->bne', q, context)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.to_out(out)
        return out + residual
