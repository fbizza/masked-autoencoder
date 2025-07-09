import numpy as np
import torch
import random
import os
import yaml
from einops import repeat
from types import SimpleNamespace

def set_seed(seed=29):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(config_name="default"):
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        all_cfgs = yaml.safe_load(f)
    cfg_dict = all_cfgs.get(config_name)
    if cfg_dict is None:
        raise ValueError(f"Config '{config_name}' not found in {config_path}")

    cfg_dict["config_name"] = config_name
    cfg_dict["model_path"] = os.path.join("data", "weights", f"{config_name}")

    return SimpleNamespace(**cfg_dict)

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

