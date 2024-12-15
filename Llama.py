from transformers.models.llama import LlamaModel, LlamaConfig
import torch

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
