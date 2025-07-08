import numpy as np
from src.utils import random_permutation_with_inverse


def test_random_permutation_with_inverse():
    length = 10
    forward, backward = random_permutation_with_inverse(length)

    print("\nForward indexes:", forward)
    print("Backward indexes:", backward)

    # check that backward is actually the inverse of forward
    recovered = forward[backward]
    print("Recovered (should be ordered):", recovered)
    assert np.array_equal(recovered, np.arange(length)), "The inversion does not work"
