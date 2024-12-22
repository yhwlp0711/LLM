from transformers.models.llama import LlamaModel, LlamaConfig
import transformers.models.llama.modeling_llama as Llama
import torch

device = torch.device('cuda')
# device = torch.device('mps')

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
    pad_token_id=0,
    use_cache=True
    )


class MyLlama(torch.nn.Module):
    def __init__(self, config):
        super(MyLlama, self).__init__()
        self.llama = LlamaModel(config)
        self.Linear = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, X, attention_mask=None):
        return self.Linear(self.llama(X, attention_mask).last_hidden_state)


def train_LlamaModel(config, device):
    model = MyLlama(config)
    model = model.to(device)
    model.train()
    X = torch.randint(low=0, high=config.vocab_size, size=(4, 100))  # (batch_size=4, len_seq=100)
    attention_mask = torch.Tensor([[1] * 90 + [0] * 10, [1] * 80 + [0] * 20, [1] * 70 + [0] * 30, [1] * 60 + [0] * 40]).reshape(4, 100)
    attention_mask = attention_mask.to(device)
    X = X.to(device)
    y_hat = model(X, attention_mask)
    print(y_hat.shape)


def test_LlamaModel(config, max_len, device):
    model = LlamaModel(config)
    model = model.to(device)
    model.eval()
    X = torch.randint(low=0, high=config.vocab_size, size=(1, 1))
    X = X.to(device)
    res = [X[0][0].item()]
    for _ in range(max_len):
        y_hat = model(X)
        y = y_hat.argmax(dim=-1).unsqueeze(0)
        res.append(y[0][0].item())
        X = y

    print(res)


# train_LlamaModel(config, device)

test_LlamaModel(config, 100, device)
