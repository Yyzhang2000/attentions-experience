from dataclasses import dataclass


@dataclass
class Config:
    attention_name: str = "mha"
    hidden_size: int = 512
    dropout: float = 0.1
    max_seq_len = 300
    vocab_size = 50257
    n_layers = 6
    num_heads: int = 8
