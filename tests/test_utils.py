import numpy as np
import torch

from src.utils import random_permutation_with_inverse, take_indices


def test_random_permutation_with_inverse():
    length = 10
    forward, backward = random_permutation_with_inverse(length)

    print("\nForward indexes:", forward)
    print("Backward indexes:", backward)

    # check that backward is actually the inverse of forward
    recovered = forward[backward]
    print("Recovered (should be ordered):", recovered)
    assert np.array_equal(recovered, np.arange(length)), "The inversion does not work"

def test_take_indices():
    # shape: (4, 2, 1)
    sequences = torch.tensor([
        [[0], [10]],
        [[1], [11]],
        [[2], [12]],
        [[3], [13]],
    ])

    # indices to reorder patches per batch, shape: (4, 2)
    indices = torch.tensor([
        [3, 2],
        [2, 3],
        [1, 1],
        [0, 0],
    ])

    output = take_indices(sequences, indices)

    print("\nInput sequences:\n", sequences.squeeze(-1))
    print("Indexes:\n", indices)
    print("Output sequences:\n", output.squeeze(-1))

    expected_batch0 = torch.tensor([3, 2, 1, 0])
    expected_batch1 = torch.tensor([12, 13, 11, 10])

    assert torch.equal(output[:, 0, 0], expected_batch0), "Batch 0 mismatch"
    assert torch.equal(output[:, 1, 0], expected_batch1), "Batch 1 mismatch"
