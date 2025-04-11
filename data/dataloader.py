import torch

import tiktoken


class DataLoader:
    def __init__(
        self, B=32, T=100, data_path="./input.txt"
    ):  # batch size  # sequence length
        self.B = B
        self.T = T

        with open(data_path, "r") as f:
            lines = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(lines)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = buf[:-1].reshape(B, T)  # input
        y = buf[1:].reshape(B, T)  # target

        self.current_position += B * T
        if self.current_position + (B * T + 1) >= len(self.tokens):
            self.current_position = 0

        return x, y

    def __len__(self):
        return len(self.tokens) // (self.B * self.T)
