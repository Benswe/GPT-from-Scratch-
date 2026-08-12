import torch

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
stoi = {ch: i for i,ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    out = []
    for c in s:
        out.append(stoi[c])
    return out

def decode(l):
    return ''.join([itos[i] for i in l])

