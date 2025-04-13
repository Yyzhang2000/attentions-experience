import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import time

from modeling.config import Config
from modeling import gpt2
from data.dataloader import DataLoader
import logging

from utils import *
from metrics import *

#### TRAINING PARAMETERS ####
TOTAL_STEPS = 5000
WARMUP_STEPS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

BATCH_SIZE = 64
SENTENCE_LENGTH = 128
DATA_PATH = "./data/train.txt"

SEED = 42
#### TRAINING PARAMETERS ####


if __name__ == "__main__":
    # Config
    config = Config(attention_name="mha")
    model_save_dir = f"logs/{config.attention_name}"
    writer = SummaryWriter(log_dir=model_save_dir)
    profiler_dir = f"{model_save_dir}/profiler"

    set_logger(os.path.join(model_save_dir, "process.log"))
    logging.info("Training Config...")

    device = get_device()
    model = gpt2.GPT2(config).to(device)

    seed_everything(SEED)
    logging.info(
        f"Model: {config.attention_name} | Batch Size: {BATCH_SIZE} | Sentence Length: {SENTENCE_LENGTH}"
    )
    logging.info(f"Total Parameters: {count_params(model) / 1e6:.2f}M")
    logging.info(
        f"Total Steps: {TOTAL_STEPS} | Warmup Steps: {WARMUP_STEPS} | Learning Rate: {LEARNING_RATE}"
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_STEPS,
        eta_min=1e-8,
    )

    dataloader = DataLoader(B=BATCH_SIZE, T=SENTENCE_LENGTH, data_path=DATA_PATH)
    logging.info("\n")

    logging.info("Training...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            # torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=20),
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

            if step % 100 == 0:
                logging.info(
                    f"[Step {step}] LR:{scheduler.get_last_lr()[0]:.8f} | Loss: {loss.item():.2f} | TPS: {tps:.2f} tokens/sec | Memory: {memory_usage:.2f} MB"
                )

    # Save the model

    save_model(model, model_save_dir, config.attention_name + "_model")
    logging.info
    (
        f"Finish Training\nModel saved to {os.path.join(model_save_dir, config.attention_name + '_model.pt')}"
    )
    writer.close()
