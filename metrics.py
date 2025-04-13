import time
import os

import psutil


def compute_tokens_per_second(batch_tokens, elapsed_time):
    if elapsed_time == 0:
        return 0.0
    return batch_tokens / elapsed_time


def compute_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert bytes to MB


METRICS_REGISTRY = {
    "tokens_per_second": compute_tokens_per_second,
    "memory_usage": compute_memory_usage,
}
