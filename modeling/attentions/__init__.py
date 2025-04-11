from .mha import MultiHeadAttention


ATTENTION_REGISTRY = {
    "mha": MultiHeadAttention,
}
