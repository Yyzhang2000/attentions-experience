import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

    def forward(self, x, kv_cache=None):
        raise NotImplementedError("Subclasses should implement this method.")
