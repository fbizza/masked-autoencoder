import numpy as np

def random_permutation_with_inverse(length: int):
    """
    Generate a random permutation of indices from 0 to length-1,
    and compute the inverse permutation that restores the original order.

    Args:
        length (int): The length of the permutation.

    Returns:
        tuple:
            - permutation (np.ndarray): shuffled indices.
            - inverse_permutation (np.ndarray): indices to invert the permutation.
    """
    ordered_indices = np.arange(length)
    permutation = np.copy(ordered_indices)
    np.random.shuffle(permutation)
    inverse_permutation = np.argsort(permutation)
    return permutation, inverse_permutation