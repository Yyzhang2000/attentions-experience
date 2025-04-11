import torch


def precompute_theta_pos_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta_base: float = 10000.0,
) -> torch.Tensor:
    assert head_dim % 2 == 0, "head_dim must be even"

    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta_base ** (theta_numerator / head_dim))

    m = torch.arange(max_seq_len).float()

    freqs = torch.outer(m, theta)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    assert x.dim() == 4, "x must be a 4D tensor, with (B, S, H, D) shape"
    assert x.shape[2] % 2 == 0, "x's last dimension must be even"

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)

    x_out = x_out.reshape(*x.shape)
    x_out = x_out.type_as(x).to(device)
    return x_out
