import torch.nn as nn
from torch import save
from torch.optim import AdamW

from data import LLMDataset, get_batches
from model.model import DecoderOnly

# Pytorch config
device = "cpu"

# Model configuration
d_model = 768
h = 64
sequence_length = 1024
vocabulary_size = 50304
batch_size = 4
nb_iteration = 1000

# debug config
nb_eval_iter = 20

# Optimizer
lr = 0.5

MODEL_CHECKPOINT_PATH = "./checkpoints"


# Get model and config
criterion = nn.CrossEntropyLoss()
model = DecoderOnly(d_model, h, vocabulary_size, sequence_length)
model.to(device)

# Get the right optimiser and loss
optimizer = AdamW(model.parameters())


def save_checkpoint(model, optimizer, loss, epoch, iteration, directory_path):
    save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        },
        f"{directory_path}/check-{iteration}-{batch}.pt",
    )


# train

for iter in range(nb_iteration):
    dataset = LLMDataset()
    for i, file in enumerate(dataset):
        batches = get_batches(sequence_length, file, batch_size)
        n = len(batches)
        for i, batch in enumerate(batches):
            x, y = batch

            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)

            out, loss = model(x, y)
            out = out.view(-1, out.size(-1))
            y = y.view(-1)

            loss = criterion(out, y)
            if i % nb_eval_iter == 0 or nb_iteration % 10 == 0:
                print(f"Iter {iter} - Batch {i} - Loss: {loss}")
                save_checkpoint(model, optimizer, loss, i, iter, MODEL_CHECKPOINT_PATH)
            loss.backward()
            optimizer.step()
