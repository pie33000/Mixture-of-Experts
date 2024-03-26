from os import listdir
from typing import Iterator

import torch
from torch.utils.data import IterableDataset
from transformers import GPT2Tokenizer


class LLMDataset(IterableDataset):
    def __init__(self, directory: str = "./data", max_length: int = 1024) -> None:
        self.directory = directory
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.files = self._create_file_iterator()

    def _create_file_iterator(self) -> Iterator:
        return iter(listdir(self.directory))

    def __iter__(self) -> Iterator:
        for file_name in self.files:
            with open(f"{self.directory}/{file_name}", "r", encoding="utf-8") as f:
                data = f.read()
            yield self.tokenizer.encode(data)


def get_batches(block_size: int, data: list[int], batch_size: int) -> torch.tensor:
    batches = []
    x, y = [], []
    for i in range(len(data) - block_size):
        if i % batch_size != 0:
            x.append(torch.tensor(data[i : i + block_size]))
            y.append(torch.tensor(data[i + 1 : i + 1 + block_size]))
        elif i > 0:
            batches.append((torch.stack(x), torch.stack(y)))
            x, y = [], []
    return batches
