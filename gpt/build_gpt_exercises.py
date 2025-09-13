"""
changes:
RELU --> GELU

Combined the Head and MultiHeadAttention into a parallel MultiHeadAttention to process heads in parallel

"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils._triton import has_triton

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile_model = has_triton()
eval_iters = 200
n_embd = 384
n_head = 6 # every head has 384/6 = 64, every head is 64
n_layer = 6
dropout = 0.2 # 20% of disabled.
# -----

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping of unique characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s:torch.tensor([stoi[c] for c in s], dtype=torch.long)
decode = lambda x:''.join([itos[i] for i in x])

# Train and Test Splits
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data with inputs x and targets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random offsets of the data
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

# estimate loss from eval_iters batches
@torch.no_grad()
def estimate_loss():
    out = {}
    compiled_m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = compiled_m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    compiled_m.train()
    return out

class ParallelMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key   = nn.Linear(n_embd, head_size * num_heads, bias = False)
        self.query = nn.Linear(n_embd, head_size * num_heads, bias = False)
        self.value = nn.Linear(n_embd, head_size * num_heads, bias = False)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("Using slow attention")
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # (batch, block_size, n_embd)
        k = self.key(x)   # (batch, block_size, head_size * num_heads)
        q = self.query(x) # (batch, block_size, head_size * num_heads)
        v = self.value(x)  # (batch, block_size, head_size * num_heads)


        k = k.view([B, T, self.num_heads, self.head_size]).transpose(1, 2)   # (batch, num_heads, block_size, head_size)
        q = q.view([B, T, self.num_heads, self.head_size]).transpose(1, 2)   # same
        v = v.view([B, T, self.num_heads, self.head_size]).transpose(1, 2)   # same

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p = dropout if self.training else 0, is_causal=True)
        else:
            wei = q @ k.transpose(-1, -2) * k.shape[-1]**-0.5 # (batch, num_heads, block_size, block_size)
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, num_heads, block_size, block_size)
            wei = F.softmax(wei, dim = -1) # (B, num_heads, T, T)

            wei = self.dropout(wei) # randomly prevent some of the nodes from communicating

            out = wei @ v # (B, num_heads, block_size, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, -1) # (B, block_size, num_heads * head_size) == (B, block_size, n_embd)

        out = self.dropout(self.proj(out))
        return out


# This is on a per token level and each token does it independently
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # in paper, the ff has dimensionality of 4 * embedding
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer going back into residual pathway
            nn.Dropout(dropout),
        )


    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = ParallelMultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # per token normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape

        # idx (4, 8)
        tok_emb = self.token_embedding_table(idx)  #(4, 8, n_embd), (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb #  (B, T, C)
        # x = self.sa_heads(x) # apply one head of self attention (B, T, C)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  #(4, 8, vocab_size), (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens (as otherwise we overflow due to above embedding tables)
            idx_cond = idx[:, -block_size: ]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device)
if compile_model and hasattr(torch, 'compile'):
    print("Compiling model...")
    compiled_m = torch.compile(m)
else:
    compiled_m = m
# print the number of parameters in the model
print(sum(p.numel() for p in compiled_m.parameters())/1e6, 'M parameters')

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(compiled_m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while eval the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evalute the loss
    logits, loss = compiled_m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
compiled_m.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(compiled_m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(compiled_m.generate(context, max_new_tokens=10000)[0].tolist()))