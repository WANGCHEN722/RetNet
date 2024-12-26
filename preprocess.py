import collections
import functools
import multiprocessing
import os
import random
from typing import Callable

import torch
from load_midi import LoadMidis


def save_processed_midi_tensor(
    num: int,
    process_func: Callable[[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    save_dir: str,
):
    data = process_func(num)
    torch.save(data, os.path.join(save_dir, f"{num}.pt"))


if __name__ == "__main__":
    print("Starting...")
    load = LoadMidis("midis", min_step=0.025)
    length = len(load)
    args = list(range(length))
    random.shuffle(args)

    func = functools.partial(
        save_processed_midi_tensor,
        process_func=load.__getitem__,
        save_dir="processed_tensors",
    )

    pool = multiprocessing.Pool()

    collections.deque(
        pool.imap_unordered(func, args, length // multiprocessing.cpu_count()), maxlen=0
    )

    pool.close()
    print("Done!")
