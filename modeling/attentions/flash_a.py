import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional

from .rope import apply_rotary_embeddings, precompute_theta_pos_frequencies
from ..config import Config
from .base import Attention
from ..kvcache import KVCache


#### Implementing Flash Attention Using Triton ####
# import triton
# import triton.language as tl


#### End of Flash Attention Implementation ####


class FlashAttention(Attention):
    def __init__(self, config: Config, layer_idx: int = 0):
        super().__init__(config, layer_idx)

        self.num_heads = config.attention_config.num_heads
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

    def forward(self, x, kv_cache: Optional[KVCache] = None):
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
        v = v.transpose(1, 2)  # [B, H, S, D]

        # If kv_cache is provided, use it for k and v
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        is_causal = S != 1
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0,
                is_causal=is_causal,
            )
        else:
            raise NotImplementedError(
                "scaled_dot_product_attention is not available in this PyTorch version."
            )

        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        attn = self.o(attn)
        attn = self.dropout(attn)

        return attn, kv_cache
