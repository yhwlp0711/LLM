import math
import torch
from torch import nn

# class SHA(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, d_model)
#         self.Wv = nn.Linear(d_model, d_model)
#         self.Wo = nn.Linear(d_model, d_model)

#         self.LN = nn.LayerNorm(d_model)


#     def forward(self, X):
#         # X: batchsize, seq_len, d_model
#         Q = self.Wq(X)
#         K = self.Wk(X)
#         V = self.Wv(X)

#         A = Q @ torch.transpose(K, -1, -2) / math.sqrt(Q.shape[-1])
#         masked = torch.tril(torch.ones(A.shape[-2], A.shape[-1])).unsqueeze(0)
#         A = A.masked_fill(masked == 0, float('-inf'))
#         A = torch.softmax(A, dim=-1)

#         O = self.Wo(A @ V)
#         return self.LN(X + O)
        

# class MHA(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, d_model)
#         self.Wv = nn.Linear(d_model, d_model)
#         self.Wo = nn.Linear(d_model, d_model)

#         self.LN = nn.LayerNorm(d_model)

#         self.Dropout = nn.Dropout(dropout)

#     def forward(self, X, masked=True):
#         batch_size, seq_len, _ = X.shape
#         Q = self.Wq(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.Wk(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.Wv(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

#         A = (Q @ torch.transpose(K, -1, -2)) / math.sqrt(self.d_k)

#         if masked:
#             mask = torch.tril(torch.ones(seq_len, seq_len))
#             print(mask)
#             A = A.masked_fill(mask == 0, float('-inf'))
#             print(A)

#         A = torch.softmax(A, -1)
#         A = self.Dropout(A)
#         O = self.Wo((A @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model))
#         return self.LN(X + self.Dropout(O))



# class MQA(nn.Module):
#     def __init__(self, d_model, heads_num, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.heads_num = heads_num
#         self.d_k = d_model // heads_num

#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, self.d_k)
#         self.Wv = nn.Linear(d_model, self.d_k)
#         self.Wo = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)
#         self.LN = nn.RMSNorm(d_model)

    
#     def forward(self, X, masked=True):
#         batch_size, seq_len, _ = X.shape

#         Q = self.Wq(X).view(batch_size, seq_len, self.heads_num, self.d_k).transpose(1, 2)
#         K = self.Wk(X)
#         K = K.unsqueeze(1).expand(batch_size, self.heads_num, seq_len, self.d_k)
#         V = self.Wv(X)
#         V = V.unsqueeze(1).expand(batch_size, self.heads_num, seq_len, self.d_k)

#         A = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)
#         if masked:
#             mask = torch.tril(torch.ones(seq_len, seq_len))
#             A = A.masked_fill(mask == 0, float('-inf'))
        
#         A = torch.softmax(A, dim=-1)
#         A = self.dropout(A)
#         O = (A @ V).transpose(1,2).contiguous().view(batch_size, seq_len, -1)
#         O = self.Wo(O)

#         return self.LN(X + self.dropout(O))
    


# class GQA(nn.Module):
#     def __init__(self, d_model, nums_head, nums_kv_head, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.nums_head = nums_head
#         self.nums_kv_head = nums_kv_head
#         self.d_k = d_model // nums_head

#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, self.d_k * nums_kv_head)
#         self.Wv = nn.Linear(d_model, self.d_k * nums_kv_head)
#         self.Wo = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)
#         self.LN = nn.RMSNorm(d_model)

    
#     def forward(self, X, masked=True):
#         batch_size, seq_len, _ = X.shape

#         Q = self.Wq(X).view(batch_size, seq_len, self.nums_head, self.d_k).transpose(1, 2)
#         K = self.Wk(X).view(batch_size, seq_len, self.nums_kv_head, self.d_k).transpose(1, 2)
#         V = self.Wv(X).view(batch_size, seq_len, self.nums_kv_head, self.d_k).transpose(1, 2)

#         K = K.repeat_interleave(self.nums_head // self.nums_kv_head, dim=1)
#         V = V.repeat_interleave(self.nums_head // self.nums_kv_head, dim=1)

#         A = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)
#         if masked:
#             mask = torch.tril(torch.ones(seq_len, seq_len))
#             A = A.masked_fill(mask==0, float('-inf'))
#         A = torch.softmax(A, dim=-1)
#         A = self.dropout(A)

#         O = (A @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

#         return self.LN(X + self.dropout(O))
    

# X = torch.randn(2, 4, 8)
# gqa = GQA(8, 4, 2)
# out = gqa(X)
# print(out)


# class MHA(nn.Module):
#     def __init__(self, num_heads, d_model, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, d_model)
#         self.Wv = nn.Linear(d_model, d_model)
#         self.Wo = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(0.1)
#         self.LN = nn.RMSNorm(d_model)

    
#     def forward(self, X, masked=True):
#         pre_X = X
#         X = self.LN(X)
#         batch_size, seq_len, _ = X.shape

#         Q = self.Wq(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
#         K = self.Wk(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
#         V = self.Wv(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        
#         A = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)
#         if masked:
#             mask = torch.tril(torch.ones(seq_len, seq_len))
#             A = A.masked_fill(mask==0, float('-inf'))
        
#         A = self.dropout(torch.softmax(A, dim=-1))

#         O = (A @ V).transpose(1,2).view(batch_size, seq_len, -1)

#         return pre_X + self.dropout(self.Wo(O))



# class MQA(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads

#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, self.d_k)
#         self.Wv = nn.Linear(d_model, self.d_k)
#         self.Wo = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)
#         self.LN = nn.RMSNorm(d_model)


#     def forward(self, X, masked=True):
#         pre_X = X
#         X = self.LN(X)
#         batch_size, seq_len, _ = X.shape

#         Q = self.Wq(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.Wk(X).unsqueeze(1).expand(batch_size,self.num_heads,seq_len,self.d_k)
#         V = self.Wv(X).unsqueeze(1).expand(batch_size,self.num_heads,seq_len,self.d_k)

#         A = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)
#         if masked:
#             mask = torch.tril(torch.ones(seq_len, seq_len))
#             A = A.masked_fill(mask==0, float('-inf'))

#         A = self.dropout(torch.softmax(A, dim=-1))

#         O = self.Wo((A @ V).transpose(1, 2).view(batch_size, seq_len, -1))

#         return pre_X + self.dropout(O)
    

# class GQA(nn.Module):
#     def __init__(self, d_model, num_heads, num_kvheads, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.num_kvheads = num_kvheads
#         self.d_k = d_model // num_heads
        
#         self.Wq = nn.Linear(d_model, d_model)
#         self.Wk = nn.Linear(d_model, num_kvheads * self.d_k)
#         self.Wv = nn.Linear(d_model, num_kvheads * self.d_k)
#         self.Wo = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)
#         self.LN = nn.RMSNorm(d_model)

    
#     def forward(self, X, masked=True):
#         pre_X = X
#         X = self.LN(X)
#         batch_size, seq_len, _ = X.shape

#         Q = self.Wq(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.Wk(X).view(batch_size, seq_len, self.num_kvheads, self.d_k).transpose(1,2)
#         K = K.repeat_interleave(self.num_heads//self.num_kvheads, dim=1)
#         V = self.Wv(X).view(batch_size, seq_len, self.num_kvheads, self.d_k).transpose(1,2)
#         V = V.repeat_interleave(self.num_heads//self.num_kvheads, dim=1)

#         A = (Q @ K.transpose(-1,-2)) / math.sqrt(self.d_k)
#         if masked:
#             mask = torch.tril(torch.ones(seq_len, seq_len))
#             A = A.masked_fill(mask==0, float('-inf'))

#         A = self.dropout(torch.softmax(A, dim=-1))

#         O = self.Wo((A @ V).transpose(1, 2).view(batch_size, seq_len, self.d_model))

#         return pre_X + self.dropout(O)

from Rope import RoPE, apply_rope

class MHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rmsnorm = nn.RMSNorm(d_model)

    
    def forward(self, X, masked=True):
        pre_X = X
        X = self.rmsnorm(X)
        batch_size, seq_len, _ = X.shape
        self.rope = RoPE(self.d_k)
        cos, sin = self.rope(seq_len)

        Q = self.Wq(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wq(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wq(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        Q, K = apply_rope(Q, K, cos, sin)

        A = (Q @ K.transpose(-1, -2)) / math.sqrt(self.d_k)
        if masked:
            mask = torch.tril(torch.ones(seq_len, seq_len))
            A = A.masked_fill(mask==0, float('-inf'))
        A = self.dropout(torch.softmax(A, dim=-1))

        O = self.Wo((A @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return pre_X + self.dropout(O)



X = torch.randn([4, 5, 12])
mha = MHA(12, 2)
print(mha(X))