from newtransformers.models.llama import LlamaModel, LlamaConfig
import newtransformers.models.llama.modeling_llama as Llama
import torch

device = torch.device('cuda')

hidden_size = 512
vocab_size = 32000

# X = torch.randint(low=0, high=vocab_size, size=(4, 100))  # (batch_size=4, len_seq=100)
# embedding = torch.nn.Embedding(vocab_size, hidden_size)
# X = embedding(X)


# def LlamaRMSNorm_test(X, hidden_size, device):
#     RMSNorm = Llama.LlamaRMSNorm(hidden_size)
#     RMSNorm = RMSNorm.to(device)
#     X = X.to(device)
#     out = RMSNorm(X)
#     # print(out.shape)
#     info = RMSNorm.extra_repr()
#     print(info)
#
#
# LlamaRMSNorm_test(X, hidden_size, device)


config = LlamaConfig(
    vocab_size=32000,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=6,
    num_attention_heads=8,
    max_position_embeddings=512,
    use_cache=False
    )

llama = LlamaModel(config)

X = torch.randint(low=0, high=32000, size=(4, 100))  # (batch_size, seq_len)

output = llama(X)
print(output)  # torch.Size([4, 100, 512])
