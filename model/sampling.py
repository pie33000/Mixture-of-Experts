import torch


def top_k_logits(logits: torch.tensor, k: int):
    # logits (d_model, vocab_size)
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k, dim=-1)
    min_values = values[:, -1].unsqueeze(-1)

    top_k_logits = torch.where(
        logits < min_values, torch.ones_like(logits) * -1e10, logits
    )
    return top_k_logits


def top_p_logit(logits, p):
    """
    Nucleus sampling
    Compute the number of logits required to be <= p
    when logits is sorted in a descending order
    """

    d_model, _ = logits.size()
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.stack(
        [
            torch.arange(0, d_model),
            # number of indices to include
            torch.clamp(torch.sum(cumulative_probs <= p, dim=-1) - 1, min=0),
        ],
        dim=-1,
    )
    min_values = torch.gather(sorted_logits, -1, indices)
    return torch.where(
        logits < min_values.unsqueeze(-1),
        torch.ones_like(logits) * -1e10,
        logits,
    )


x = top_p_logit(torch.randn((768, 1000)), 0.1)
x
