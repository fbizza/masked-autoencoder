import numpy as np
import torch
from einops import repeat

def random_permutation_with_inverse(length: int):
    # to generate indices and invert them
    ordered_indices = np.arange(length)
    permutation = np.copy(ordered_indices)
    np.random.shuffle(permutation)
    inverse_permutation = np.argsort(permutation)
    return permutation, inverse_permutation

def take_indices(sequences: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # takes patches based on indices across batches
    return torch.gather(sequences, 0, repeat(indices, 'p b -> p b c', c=sequences.shape[-1]))