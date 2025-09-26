import torch
from torch import nn
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_position_embeddings)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)  # (max_seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_emb", emb.cos()[None, :, :], persistent=False)  # (1, max_seq_len, dim)
        self.register_buffer("sin_emb", emb.sin()[None, :, :], persistent=False)  # (1, max_seq_len, dim)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        return (self.cos_emb[:, :seq_len, :], self.sin_emb[:, :seq_len, :])

def apply_rope(q, k, cos, sin, position_ids):
    pass

class MLAConfig():
    hidden_size: int
    num_heads: int
    max_seq_len: int
    rope_theta: float

    attention_dropout: float

    q_lora_rank: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int

    v_head_dim: int



class MLA(nn.Module):
    def __init__(self, config:MLAConfig):
        super().__init__()

        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        self.Wo = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size)

        
        # 压缩
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        # 到c_q
        self.q_down = nn.Linear(self.hidden_size, self.q_lora_rank)
        self.q_down_norm = nn.RMSNorm(self.q_lora_rank)
        # 到c_kv + k_rope, 后续要做split
        self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank + config.qk_rope_head_dim)
        self.kv_down_norm = nn.RMSNorm(self.kv_lora_rank)
        # 升维
        # 到nope + rope, 后续要做split
        self.q_up = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim)
        self.kv_up = nn.Linear(self.kv_lora_rank, self.num_heads * (config.qk_nope_head_dim + self.v_head_dim))


        # RoPE
        self.rope = RotaryEmbedding(dim=config.qk_rope_head_dim, max_position_embeddings=config.max_seq_len, base=config.rope_theta)


    
    def forward(self, hidden_states, position_ids, masked=None):
        batch_size, seq_len, _ = hidden_states.shape

        # 压缩
        q = self.q_down(hidden_states)
        q = self.q_down_norm(q)
        # 升维
        q = self.q_up(q) # q: (b, s, num_heads * q_head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)


        # 压缩
        c_kv = self.kv_down(hidden_states)
        ckv, k_rope= torch.split(c_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # k_rope shape: (b, s, qk_rope_head_dim)
        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_up(self.kv_down_norm(c_kv))
        # kv shape: (b, s, qk_nope_head_dim + v_head_dim)
        kv = kv.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)

        k_nope, v_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)


        # RoPE
        cos, sin = self.rope(seq_len)
        q_rope, k_rope = apply_rope(q_rope, k_rope, cos, sin, position_ids)

        query_states = torch.concat([q_nope, q_rope],dim=-1)
        key_states = torch.concat([k_nope, k_rope], dim=-1)

        A = (query_states @ key_states.transpose(-1,-2)) / math.sqrt(self.q_head_dim)
        
        if masked:
            mask = torch.tril(torch.ones(seq_len, seq_len))
            A = A.masked_fill(mask==0, float('-inf'))
        
        A = torch.softmax(A,dim=-1)
        A = torch.dropout(A, self.attention_dropout, train=self.training)
        O = self.Wo((A @ v_states).transpose(1, 2).view(batch_size, seq_len, -1))

        return hidden_states + O

