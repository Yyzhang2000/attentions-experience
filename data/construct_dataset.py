import torch
import numpy as np

from tqdm import tqdm

from hftokenizer import HFTokenizer


def construct_dataset(data_file, sequence_length=256):
    """
    Constructs a dataset from the given data file.

    Args:
        data_file (str): Path to the data file.
        sequence_length (int): Length of each sequence.

    Returns:
        torch.Tensor: Constructed dataset as a tensor.
    """
    tokenizer = HFTokenizer()
    tokenizer.load()

    print("Loading data...")
    with open(data_file, "r") as f:
        lines = f.readlines()

    dataset = []
    packed_sequence = []

    sample_idx = 0
    i = 0

    while sample_idx < len(lines):
        if sample_idx % 10_000 == 0:
            print(f"{sample_idx/len(lines)}")

        # Tokenize sample
        f = lines[sample_idx]
        t = tokenizer.encode(f)

        t_len = len(t)
        current_len = len(packed_sequence)

        # pack to packed sequence until sequence is packed or sample ends
        while current_len < sequence_length and i < t_len:
            packed_sequence.append(t[i])
            i += 1
            current_len += 1

        # if sample ends, go to next sample
        if i == t_len:
            sample_idx += 1
            i = 0
        # if sequence is packed, append + pack new sequence
        # note: we don't reset i here since we need to pick up where we left off in the sample
        # note note: this is an if, not an elif, in case they happen at the same time
        if current_len == sequence_length:
            # eos token
            packed_sequence.append(0)
            dataset.append(packed_sequence)
            packed_sequence = []

    # save as shuffled np array
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    print(np.shape(dataset))

    with open("./packed_data.npy", "wb") as f:
        np.save(f, dataset)


if __name__ == "__main__":
    construct_dataset("./data.txt", 1024)
    print("Dataset constructed.")
