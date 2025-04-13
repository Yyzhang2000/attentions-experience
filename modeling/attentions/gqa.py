import torch
import torch.nn as nn

from typing import Optional

from .rope import apply_rotary_embeddings, precompute_theta_pos_frequencies
from ..config import Config
from .base import Attention
from ..kvcache import KVCache


def repeat_kv(k, v, n_rep):
    """Repeat k and v for each head in the group"""
    B, H, S, D = k.shape
    if n_rep == 1:
        return k, v
    # Repeat k and v for each head in the group
    k = k[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)
    v = v[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)
    return k, v


class GroupQueryAttention(Attention):
    def __init__(self, config: Config, layer_idx: int = 0):
        super().__init__(config, layer_idx)

        self.num_heads = config.attention_config.num_heads
        self.hidden_size = config.hidden_size
        assert (
            self.hidden_size % self.num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.head_dim = self.hidden_size // self.num_heads

        # self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
        self.num_kv_heads = config.attention_config.num_kv_heads
        self.q = nn.Linear(self.hidden_size, self.head_dim * self.num_heads)
        # self.kv = nn.Linear(self.hidden_size, 2 * self.hidden_size)
        self.k = nn.Linear(self.hidden_size, self.head_dim * self.num_kv_heads)
        self.v = nn.Linear(self.hidden_size, self.head_dim * self.num_kv_heads)

        self.n_rep = self.num_heads // self.num_kv_heads
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"

        self.o = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        self.max_seq_len = config.max_seq_len

        self.layer_idx = layer_idx

        theta_pos_freq = precompute_theta_pos_frequencies(
            self.head_dim, self.max_seq_len * 2
        )
        self.register_buffer("theta_pos_freq", theta_pos_freq, persistent=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)) == 1,
            persistent=False,
        )

    def forward(self, x, kv_cache: Optional[KVCache] = None):
        B, S, D = x.shape

        q = self.q(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(B, S, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        freqs_complex = self.theta_pos_freq[:S].to(x.device)
        q = apply_rotary_embeddings(q, freqs_complex, x.device)
        k = apply_rotary_embeddings(k, freqs_complex, x.device)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # [B, H, S, D]

        k, v = repeat_kv(k, v, self.n_rep)

        # If kv_cache is provided, use it for k and v
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        print(k.shape, q.shape, v.shape)
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if q.shape[-2] == k.shape[-2]:  # In the prefilling stage. or not use KV Cache
            # If S not 1, we need to apply the causal mask
            scores = scores.masked_fill(
                self.causal_mask[:S, :S].to(x.device) == 0, float("-inf")
            )

        attn_weights = scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        attn = self.o(attn)
        attn = self.dropout(attn)

        return attn, kv_cache
