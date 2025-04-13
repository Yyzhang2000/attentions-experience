from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Type


@dataclass
class AttentionConfigBase:
    """Base class for attention-specific config"""

    num_heads: int = 12


@dataclass
class GQAConfig(AttentionConfigBase):
    num_kv_heads: int = 4


ATTENTION_CONFIG_MAP: Dict[str, Type[AttentionConfigBase]] = {
    "mha": AttentionConfigBase,
    "gqa": GQAConfig,
}


@dataclass
class Config:
    attention_name: str = "mha"
    hidden_size: int = 768
    dropout: float = 0.1
    max_seq_len: int = 300
    vocab_size: int = 50257
    n_layers: int = 12
    attention_config: AttentionConfigBase = field(
        default_factory=AttentionConfigBase
    )  # extendable

    def __init__(self, attention_name: str = "mha", **kwargs):
        self.attention_name = attention_name

        config_cls = ATTENTION_CONFIG_MAP.get(attention_name)
        if config_cls is not None:
            self.attention_config = config_cls(**kwargs)
        else:
            raise ValueError(f"Unknown attention name: {attention_name}")
