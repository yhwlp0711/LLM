import torch
from torch import nn
import math

class LoraLayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, merge, dropout=0.1):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        self.linear = nn.Linear(in_features, out_features)

        if rank > 0:
            self.linear.weight.requires_grad = False  # 冻结原始权重
            self.lora_a = nn.Parameter(torch.randn(rank, in_features))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))

            self.scale = alpha / rank

        self.dropout = nn.Dropout(dropout)

        if merge:
            self.merge_weight()


    def merge_weight(self):
        if self.rank > 0:
            weight = self.linear.weight.data
            weight += self.scale * (self.lora_b @ self.lora_a)
            self.linear.weight.data = weight

    
    def forward(self, X):
        if self.rank > 0:
            output_part1 = self.linear(X)
            output_part2 = self.scale * (X @ (self.lora_b @ self.lora_a).T)
            output = output_part1 + output_part2
        else:
            output = self.linear(X)
        return self.dropout(output)
    

X = torch.randn(2, 4)
lora = LoraLayer(4, 8, rank=2, alpha=16, merge=False)
out = lora(X)
print(out.shape)