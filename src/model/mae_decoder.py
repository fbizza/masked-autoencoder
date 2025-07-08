import torch

from einops import rearrange
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block

from src.utils import take_indices


class Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layers=4,
                 num_heads=3,
                 ) -> None:
        super().__init__()

        self.num_patches = (image_size // patch_size) ** 2
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.image_size = image_size

        self.pos_embedding = torch.nn.Parameter(torch.zeros(self.num_patches + 1, 1, emb_dim))  # (+1 for cls token)

        self.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, emb_dim))  # (because masked patches don't have embedding from the encoder)

        self.transformer_blocks = torch.nn.Sequential(*[
            Block(emb_dim, num_heads) for _ in range(num_layers)
        ])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)

        self.unpatchify = Rearrange(
            '(h w) b (c p1 p2) -> b c (h p1) (w p2)',
            p1=patch_size,
            p2=patch_size,
            h=image_size // patch_size
        )

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        num_visible_patches_plus_cls = features.shape[0]  # (e.g 65)
        batch_size = features.shape[1]
        cls_index = torch.zeros(1, batch_size, dtype=backward_indexes.dtype, device=backward_indexes.device)
        backward_indexes_shifted = backward_indexes + 1  # shift to add cls token at the beginning
        all_indices = torch.cat([cls_index, backward_indexes_shifted], dim=0)  # [257, 2]
        num_masked_patches = all_indices.shape[0] - num_visible_patches_plus_cls  # 192
        mask_tokens_expanded = self.mask_token.expand(num_masked_patches, batch_size, self.emb_dim)
        features_complete = torch.cat([features, mask_tokens_expanded], dim=0)  # [257, 2, 192]
        features_ordered = take_indices(features_complete, all_indices)
        features_pos = features_ordered + self.pos_embedding
        features_batch_first = rearrange(features_pos, 't b c -> b t c')  # t=257

        transformed = self.transformer_blocks(features_batch_first)

        transformed = rearrange(transformed, 'b t c -> t b c')
        transformed_patches = transformed[1:]  # remove cls token

        patches_pixels = self.head(transformed_patches)  # [256, 2, 12]

        mask_tensor = torch.zeros_like(patches_pixels)
        mask_start_idx = num_visible_patches_plus_cls - 1
        mask_tensor[mask_start_idx:] = 1  # 0=visible 1=masked
        mask_ordered = take_indices(mask_tensor, backward_indexes)

        reconstructed_img = self.unpatchify(patches_pixels)
        reconstructed_mask = self.unpatchify(mask_ordered)

        return reconstructed_img, reconstructed_mask