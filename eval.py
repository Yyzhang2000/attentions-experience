import torch
from modeling.config import Config
from modeling import gpt2
import tiktoken
import argparse
from modeling.config import Config
from modeling import gpt2
from data.dataloader import DataLoader
from modeling.kvcache import KVCache

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
SENTENCE_LENGTH = 128
DATA_PATH = "./data/eval.txt"


def load_model(model_path, config):
    model = gpt2.GPT2(config)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


import time
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


def generate(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    input_ids = (
        torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(DEVICE)
    )  # shape: [1, T]
    kv_cache = KVCache()
    generated = input_ids

    with torch.no_grad():
        start_time = time.time()
        for _ in range(max_new_tokens):
            # Pass past_key_values to enable KV caching
            logits, loss, kv_cache = model(input_ids, kv_cache=kv_cache)
            # logits, _ = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = next_token_id  # 0.7767 seconds
            # input_ids = torch.cat((input_ids, next_token_id), dim=-1)  # 1.6501 seconds
            generated = torch.cat([generated, next_token_id], dim=-1)

    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.4f} seconds")
    return tokenizer.decode(generated.squeeze(0).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Is pale cold cowardice")
    parser.add_argument("--model_path", type=str, default="logs/mha/mha_model.pt")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()

    config = Config(attention_name="mha")
    tokenizer = tiktoken.get_encoding("gpt2")
    model = load_model(args.model_path, config)

    output = generate(model, tokenizer, args.prompt, max_new_tokens=args.tokens)
    print("Generated text:\n\n" + output)
