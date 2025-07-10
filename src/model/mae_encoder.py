import torch
from einops import rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

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


class Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layers=12,
                 num_heads=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        num_patches = (image_size // patch_size) ** 2

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))

        self.pos_embedding = torch.nn.Parameter(torch.zeros(num_patches, 1, emb_dim))  # NOTE: learnable positional embedding as in the orignal ViT

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)  # (Convolution used for simplicity, it would be the same as usign extraction + flatten + linear)

        self.shuffle = MaskPatches(mask_ratio)

        self.transformer_blocks = torch.nn.Sequential(*[
            Block(emb_dim, num_heads) for _ in range(num_layers)
        ])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        # (init of transformer blocks is handled by the library following the original ViT)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)  # e.g. [2, 3, 32, 32] -> [2, 192, 16, 16]

        patches = rearrange(patches, 'b c h w -> (h w) b c')  # e.g. [2, 192, 16, 16] -> [256, 2, 192]

        patches = patches + self.pos_embedding  # still [256, 2, 192]

        patches, forward_indices, backward_indices = self.shuffle(
            patches)  # [256, 2, 192] -> [64, 2, 192] (e.g. removes 75% of patches), indices have shape [256, 2] ([num_patches, B])

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches],
                            dim=0)  # [64, 2, 192] -> [65, 2, 192]

        patches = rearrange(patches, 't b c -> b t c')  # [65, 2, 192] -> [2, 65, 192]

        features = self.layer_norm(self.transformer_blocks(patches))  # [2, 65, 192] -> [2, 65, 192]

        features = rearrange(features, 'b t c -> t b c')  # [2, 65, 192] -> [65, 2, 192]

        return features, backward_indices