"""Tiny causal transformer that consumes the packed-row contract directly (kept from the earlier prototype).

It takes token ids, position ids and segment ids; the attention mask is built
from segment ids exactly as packing.attention_mask defines it, so a bug in
segment ids changes what the model can see (tested in test_packing.py).
Runs on CPU with deterministic kernels (see trainer.setup_determinism).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hashing import sha256_bytes


def build_attn_mask(seg):
    """seg: [B, L] long -> bool [B, L, L], True = may attend."""
    L = seg.shape[1]
    same = seg[:, :, None] == seg[:, None, :]
    real = seg[:, :, None] > 0
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    m = same & real & causal
    eye = torch.eye(L, dtype=torch.bool).expand_as(m)
    return m | eye


class Block(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, L, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // self.h)
        att = att.masked_fill(~mask[:, None, :, :], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x))


class TinyGPT(nn.Module):
    """Output head tied to the token embedding (keeps checkpoints small with A2's 10K vocabulary)."""

    def __init__(self, vocab, max_len, d_model, n_layers, n_heads, d_ff, **_):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def head(self, h):
        return h @ self.tok.weight.t()

    def hidden(self, tokens, positions, seg):
        x = self.tok(tokens) + self.pos(positions)
        mask = build_attn_mask(seg)
        for b in self.blocks:
            x = b(x, mask)
        return self.ln_f(x)

    def forward(self, tokens, positions, seg):
        return self.head(self.hidden(tokens, positions, seg))


def token_losses(model, batch):
    """Per-position cross entropy [B, L] (0 where labels are IGNORE)."""
    logits = model(batch["tokens"], batch["position_ids"], batch["segment_ids"])
    lab = batch["labels"]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.clamp(min=0).reshape(-1), reduction="none")
    return ce.view(lab.shape) * (lab >= 0)


def state_hash(state_dict) -> str:
    parts = []
    for k in sorted(state_dict):
        v = state_dict[k]
        if torch.is_tensor(v):
            a = v.detach().cpu().contiguous().numpy()
            parts.append(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        else:
            parts.append(k.encode() + repr(v).encode())
    return sha256_bytes(b"|".join(parts))


def optimizer_hash(opt) -> str:
    sd = opt.state_dict()
    flat = {}
    for pid, st in sd["state"].items():
        for k, v in st.items():
            flat[f"{pid}.{k}"] = v if torch.is_tensor(v) else torch.tensor(v)
    flat["param_groups"] = torch.tensor(np.array([[g["lr"], g["weight_decay"]] for g in sd["param_groups"]]))
    return state_hash(flat)
