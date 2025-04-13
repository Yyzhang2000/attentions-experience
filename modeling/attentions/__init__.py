from .mha import MultiHeadAttention
from .gqa import GroupQueryAttention


ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
    "gqa": GroupQueryAttention,
}
