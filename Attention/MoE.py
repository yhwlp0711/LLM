import torch
from torch import nn


class Expert(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class MoE(nn.Module):
    def __init__(self, expert_dim, hidden_size, num_experts, top_k):
        super().__init__()

        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(hidden_size, expert_dim, hidden_size) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_size, num_experts)

    
    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        num_tokens = batch_size * seq_len
        X_flat = X.view(num_tokens, -1)
        top_weights, top_idxs = torch.topk(torch.softmax(self.gate(X_flat), dim=-1), self.top_k)
        top_weights = top_weights / torch.sum(top_weights, dim=-1, keepdim=True)
        output = torch.zeros_like(X_flat)
        for i in range(num_tokens):
            for idx in range(self.top_k):
                expert = self.experts[top_idxs[i, idx]]
                output[i] += top_weights[i, idx] * expert(X_flat[i])
        
        output = output.view(batch_size, seq_len, -1)
    
        return output

X = torch.randn(2, 4, 16)
moe = MoE(expert_dim=8, hidden_size=16, num_experts=4, top_k=2)
out = moe(X)
print(out.shape)