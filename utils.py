import torch
import os
import numpy as np


def save_model(model, model_dir, model_name):
    """
    Save the model to a file.

    Args:
        model (torch.nn.Module): The model to save.
        model_name (str): The name of the model file.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, model_name + ".pt"))
    print(f"Model saved to {os.path.join(model_dir, model_name + '.pt')}")


def get_device():
    """
    Get the device to use for PyTorch operations.

    Returns:
        torch.device: The device to use (CPU or GPU).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_model(model, model_dir, model_name):
    """
    Load the model from a file.

    Args:
        model (torch.nn.Module): The model to load.
        model_name (str): The name of the model file.
    """
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name + ".pt")))
    print(f"Model loaded from {os.path.join(model_dir, model_name + '.pt')}")


def count_params(model):
    """
    Count the number of parameters in the model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_tensor_memory_MB(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / 1024**2


def seed_everything(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
