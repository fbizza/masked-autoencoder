import torch
from src.utils import random_permutation_with_inverse, take_indices

class MaskPatches(torch.nn.Module):
    def __init__(self, masking_ratio) -> None:
        super().__init__()
        self.masking_ratio = masking_ratio

    def forward(self, patches: torch.Tensor):
        num_patches, batch_size, embed_dim = patches.shape
        num_keep = int(num_patches * (1 - self.masking_ratio))

        permutations = [random_permutation_with_inverse(num_patches) for _ in range(batch_size)]

        permutation_indices = torch.stack([
            torch.tensor(p[0], dtype=torch.long, device=patches.device) for p in permutations
        ], dim=-1)

        inverse_permutation_indices = torch.stack([
            torch.tensor(p[1], dtype=torch.long, device=patches.device) for p in permutations
        ], dim=-1)

        patches = take_indices(patches, permutation_indices)
        patches = patches[:num_keep]  # NOTE: as described in the paper they simply cut the extra patches, they don't use any [MASK] special token

        return patches, permutation_indices, inverse_permutation_indices