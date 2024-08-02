# Transformer-Mixture-of-Experts (TMOE)
==========================

A PyTorch implementation from scratch of the Transformer architecture and Mixture of Experts model.

## Overview
---------------

This project aims to implement the Transformer architecture [1] and Mixture of Experts (MoE) model [2] from scratch in PyTorch. The Transformer architecture is a sequence-to-sequence model that has achieved state-of-the-art results in machine translation, question answering, and other tasks. MoE, on the other hand, is a neural network framework that learns to select the best expert among multiple candidates.

## Architecture
----------------

The TMOE model consists of two main components:

1. **Transformer**: A sequence-to-sequence Transformer encoder-decoder architecture.
2. **Mixture of Experts**: A MoE module that selects the best expert among multiple candidates based on the input features.

### Transformer

The Transformer architecture is a multi-head attention mechanism with self-attention and feed-forward neural networks.

*   Self-Attention Mechanism: The self-attention mechanism allows the model to attend to different parts of the input sequence simultaneously.
*   Feed-Forward Neural Networks (FFNNs): FFNNs are used as position-wise affine transformations.

### Mixture of Experts

The MoE module selects the best expert among multiple candidates based on the input features. The experts are trained separately and then combined using a gating mechanism.

## Implementation
------------------

This project is implemented in PyTorch 2.2.0. The code is well-documented and follows PEP 8 guidelines for naming conventions and coding style.

### Training Code

The training code is available in the `train.py` file. It uses the `torch.optim.Adam` optimizer with a learning rate of 0.001 and a batch size of 32.


## Usage
---------

To run the code, simply execute the following command:

```bash
python train.py
```

This will start the training process and save the results to the `results.txt` file.

## License
------------

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## References
--------------

[1] Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762.

[2] Mixture of Experts Explained [arXiv:1710.05155.](https://huggingface.co/blog/moe)

