import torch
import torch.nn as nn

from .rope import apply_rotary_embeddings, precompute_theta_pos_frequencies
from ..config import Config
from .base import Attention


class MultiHeadAttention(Attention):
    def __init__(self, config: Config, layer_idx: int = 0):
        super().__init__(config, layer_idx)

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        assert (
            self.hidden_size % self.num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv = nn.Linear(self.hidden_size, 3 * self.hidden_size)
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

    def forward(self, x, kv_cache=None, pos=None):
        B, S, D = x.shape

        q, k, v = map(
            lambda t: t.reshape(B, S, self.num_heads, self.head_dim),
            self.qkv(x).chunk(3, dim=-1),
        )

        # Apply rotary embeddings
        freqs_complex = self.theta_pos_freq[:S].to(x.device)
        q = apply_rotary_embeddings(q, freqs_complex, x.device)
        k = apply_rotary_embeddings(k, freqs_complex, x.device)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # If kv_cache is provided, use it for k and v
        if kv_cache is not None and pos is not None:
            k = torch.cat([kv_cache, k], dim=2)
            v = torch.cat([kv_cache, v], dim=2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        scores = scores.masked_fill(
            self.causal_mask[:S, :S].to(x.device) == 0, float("-inf")
        )

        attn_weights = scores.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn = torch.matmul(attn_weights, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        attn = self.o(attn)
        attn = self.dropout(attn)

        return attn
