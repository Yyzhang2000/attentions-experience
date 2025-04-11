import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .attentions import ATTENTION_REGISTRY


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.ln1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.ln2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.ln2(self.dropout(F.gelu(self.ln1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = ATTENTION_REGISTRY[config.attention_name](config, layer_idx)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

    def forward(self, x):
        return self.embedding(x)


class ProjectorHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.projector = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x):
        return self.projector(x)


class GPT2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config, i) for i in range(config.n_layers)]
        )
        self.projector = ProjectorHead(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # Weights tieing
        self.projector.weight = self.embedding.embedding.weight

    def forward(self, ids: torch.Tensor, targets=None):
        B, S = ids.shape

        assert S <= self.config.max_seq_len, "Sequence length exceeds maximum length"

        x = self.embedding(ids)

        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        logits = self.projector(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
