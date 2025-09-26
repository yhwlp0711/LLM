import torch
from torch import nn

# class Rope(nn.Module):
#     def __init__(self, d_k, base=1e5):
#         super().__init__()
#         self.base = base
#         self.d_k = d_k

#         self.inv_freq = 1.0/(base**(torch.arange(0, d_k, 2, dtype=torch.int64).float()/d_k))

#     def forward(self, X):
#         # X: b, s, d
#         # inv: d_k/2
#         batch_size, seq_len, _ = X.shape
#         position_ids = torch.arange(seq_len, dtype=torch.long, device=X.device).unsqueeze(0)  # b, s
#         # b, d_k/2, 1
#         inv_freq_expanded = self.inv_freq[None,:,None].expand(batch_size,-1,1)
#         # b, 1, s
#         position_ids_expanded = position_ids[:,None,:]

#         # b, d_k/2, s -> b, s, d_k/2
#         freqs = (inv_freq_expanded * position_ids_expanded).transpose(1,2)
#         emb = torch.cat((freqs, freqs), dim=-1)

#         return emb.cos(), emb.sin()


# def rotate_half(x):
#     x1 = x[..., :x.shape[-1]//2]
#     x2 = x[..., x.shape[-1]//2:]
#     return torch.cat((-x2, x1), dim=-1)

# def apply(q, k, cos, sin):
#     cos = cos.unsqueeze(dim=1)
#     sin = sin.unsqueeze(dim=1)

#     q_embed = (q * cos) + (rotate_half(q) * sin)  # 广播到 (batch_size, num_heads, seq_len, head_dim)  即每个头都用
#     k_embed = (k * cos) + (rotate_half(k) * sin)

#     return q_embed, k_embed


class RoPE(nn.Module):
    def __init__(self, d_k, base=1e5):
        super().__init__()
        self.d_k = d_k
        self.base = base

        self.inv_freq = 1.0 / self.base ** (torch.arange(0, d_k, 2)/d_k) # (d_k / 2)
    
    def forward(self, seq_len):
        # (d_k / 2) -> (d_k/2, 1)
        inv_freq_expand = self.inv_freq[None, :]
        position_ids = torch.arange(0, seq_len) # (s)
        # (s) -> (1, s)
        position_ids_expand = position_ids[:, None]

        freqs = (inv_freq_expand * position_ids_expand) # (s, d_k/2)
        emb = torch.cat([freqs, freqs], dim=-1) # (s, d_k)

        return emb.cos(), emb.sin()

def rotary_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]  # (1, 1, s, d_k)
    sin = sin[None, None, :, :]  # (1, 1, s, d_k)
    q_emb = q * cos + rotary_half(q) * sin
    k_emb = k * cos + rotary_half(k) * sin

    return q_emb, k_emb
