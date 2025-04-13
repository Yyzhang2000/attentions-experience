from .mha import MultiHeadAttention
from .gqa import GroupQueryAttention
from .flash_a import FlashAttention


ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "gqa": GroupQueryAttention,
    "flash_a": FlashAttention,
}
