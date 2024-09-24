# Copyright 2024 The ingestables Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IngesTables built-in aligners.

Each modality may have a different representation -- different dimensionality,
different ranks, etc. But in order to use transformers to attend across features
of different modalities, each feature needs to be projected to the same shape
so that it can be stacked with other features. These aligners define how each
modality is to be projected to a common shape.
"""

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass
class CatAlignerConfig:
  x_key_dim: int  # Input dimension.
  x_val_dim: int  # Input dimension.
  z_key_dim: int  # Hidden dimension.
  z_val_dim: int  # Hidden dimension.


class CatAligner(nn.Module):
  """Alignment module to pass categorical features to ingestables.Encoder."""

  def __init__(self, config: CatAlignerConfig):
    super().__init__()
    self.key_align = nn.Linear(
        in_features=config.x_key_dim,
        out_features=config.z_key_dim,
        bias=False,
    )
    self.val_align = nn.Linear(
        in_features=config.x_val_dim,
        out_features=config.z_val_dim,
        bias=False,
    )

  def forward(
      self,
      x_keys: torch.Tensor,
      x_vals: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed a categorical cell before feeding into backbone.

    Args:
      x_keys: [..., x_key_dim] float tensor.
      x_vals: [..., x_val_dim] float tensor.

    Returns:
      Tuple of ([..., z_key_dim], [..., z_val_dim]) float tensors.
    """
    z_key_emb = self.key_align(x_keys)
    z_val_emb = self.val_align(x_vals)
    return z_key_emb, z_val_emb


@dataclasses.dataclass
class NumAlignerConfig:
  x_key_dim: int  # Input dimension.
  x_val_dim: int  # Input dimension.
  z_key_dim: int  # Hidden dimension.
  z_val_dim: int  # Hidden dimension.


class NumAligner(nn.Module):
  """Alignment module to pass numerical features to ingestables.Encoder."""

  def __init__(self, config: NumAlignerConfig):
    super().__init__()
    self.key_align = nn.Linear(
        in_features=config.x_key_dim,
        out_features=config.z_key_dim,
        bias=False,
    )
    self.val_align = nn.Linear(
        in_features=config.x_val_dim,
        out_features=config.z_val_dim,
        bias=False,
    )

  def forward(
      self,
      x_keys: torch.Tensor,
      x_vals: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed a numerical cell before feeding into backbone.

    Args:
      x_keys: [..., x_key_dim] float tensor.
      x_vals: [..., x_val_dim] float tensor.

    Returns:
      Tuple of ([..., z_key_dim], [..., z_val_dim]) float tensors.
    """
    z_key_emb = self.key_align(x_keys)
    z_val_emb = self.val_align(x_vals)
    return z_key_emb, z_val_emb
