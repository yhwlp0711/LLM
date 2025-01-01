import torch
from torch import nn

from transformers.models.qwen2 import Qwen2Model, Qwen2Config

# device = torch.device('cuda')
device = torch.device('mps')

hidden_size = 768
vocab_size = 32000

Config = Qwen2Config(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    intermediate_size=2048,
    num_attention_heads=8,
    num_hidden_layers=6,
    max_position_embeddings=512,
    _attn_implementation='eager',
    use_cache=True
)


class MyQwen2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qwen2 = Qwen2Model(config)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, x, attention_mask=None, past_key_values=None):
        out = self.qwen2(x, attention_mask=attention_mask, past_key_values=past_key_values)
        return self.linear(out)
    


def train_model(config, device):
    net = MyQwen2(config)
    net.to(device)
    X = torch.randint(low=0, high=config.vocab_size, size=(4, 100))
    X = X.to(device)
    attention_mask = torch.Tensor([[1] * 90 + [0] * 10, [1] * 80 + [0] * 20, [1] * 70 + [0] * 30, [1] * 60 + [0] * 40]).reshape(4, 100)
    attention_mask = attention_mask.to(device)
    y_hat = net(X, attention_mask=attention_mask)
    print(y_hat.shape)


def test_model(config, max_len, device):
    net = MyQwen2(config)
    net = net.to(device)
    X = torch.randint(low=0, high=config.vocab_size, size=(1,1))
    past_key_values = None
    res = [X[0][0].item()]
    for _ in range(max_len):
        y_hat, past_key_values = net(X, past_key_values=past_key_values)
        y = y_hat.argmax(dim=-1)
        res.append(y[0][0].item())
        X = y
    print(res)


train_model(Config, device)