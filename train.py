import torch.nn as nn
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

# debug config
nb_eval_iter = 20

# Optimizer
lr = 0.5


# Get model and config
criterion = nn.CrossEntropyLoss()
model = DecoderOnly(d_model, h, vocabulary_size, sequence_length)
model.to(device)

# Get the right optimiser and loss
optimizer = AdamW(model.parameters())


# train

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
        if i % nb_eval_iter == 0:
            print(f"Batch {i} - Loss: {loss}")
        loss.backward()
        optimizer.step()
