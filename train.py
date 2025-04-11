import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

from modeling.config import Config
from modeling import gpt2
from data.dataloader import DataLoader

from utils import *
from metrics import *

#### TRAINING PARAMETERS ####
TOTAL_STEPS = 1000
WARMUP_STEPS = 50
LEARNING_RATE = 1e-4

BATCH_SIZE = 32
SENTENCE_LENGTH = 128
DATA_PATH = "./data/input.txt"

SEED = 42
#### TRAINING PARAMETERS ####


if __name__ == "__main__":
    # Config
    config = Config(attention_name="mha")
    device = torch.device("mps")

    model = gpt2.GPT2(config).to(device)
    model_save_dir = f"logs/{config.attention_name}"
    writer = SummaryWriter(log_dir=model_save_dir)
    profiler_dir = f"{model_save_dir}/profiler"

    seed_everything(SEED)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = cosine_with_warmup_lr_scheduler(
        optimizer, total_steps=TOTAL_STEPS, warmup_steps=WARMUP_STEPS
    )

    dataloader = DataLoader(B=BATCH_SIZE, T=SENTENCE_LENGTH, data_path=DATA_PATH)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            # Works for M1/M2 GPUs
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        for step in range(TOTAL_STEPS):
            start_time = time.time()

            x, y = dataloader.next_batch()
            x = x.to(torch.int64).to(device)
            y = y.to(torch.int64).to(device)

            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            elapsed_time = time.time() - start_time
            tokens_processed = (
                x.numel()
            )  # Assuming x is a tensor of shape (batch_size, seq_len)
            tps = compute_tokens_per_second(tokens_processed, elapsed_time)
            # Assuming you have a function to compute memory usage
            memory_usage = compute_memory_usage()

            profiler.step()

            writer.add_scalar("MemoryUsage/train", memory_usage, step)
            writer.add_scalar("TokensPerSec/train", tps, step)
            writer.add_scalar("Loss/train", loss.item(), step)

            if step % 10 == 0:
                print(f"Step {step}: loss = {loss.item()}")

    # Save the model
    save_model(model, model_save_dir, config.attention_name + "_model")
    writer.close()
