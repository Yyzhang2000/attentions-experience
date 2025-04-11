import torch
import os
import numpy as np


def cosine_with_warmup_lr_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
):
    """
    Cosine annealing with warmup learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to apply the scheduler to.
        total_steps (int): Total number of training steps.
        warmup_steps (int): Number of warmup steps.
    """

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-8,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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


def load_model(model, model_dir, model_name):
    """
    Load the model from a file.

    Args:
        model (torch.nn.Module): The model to load.
        model_name (str): The name of the model file.
    """
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name + ".pt")))
    print(f"Model loaded from {os.path.join(model_dir, model_name + '.pt')}")


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
