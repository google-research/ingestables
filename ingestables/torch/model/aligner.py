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

from typing import Literal, Optional

from ingestables.torch.model.lib import categorical_embeddings as catlib
from ingestables.torch.model.lib import numerical_embeddings as numlib
import torch
from torch import nn


class TextualAligner(nn.Module):
  """Alignment module to pass categorical and string features to ingestables encoder."""

  def __init__(
      self,
      x_key_dim: int,
      x_val_dim: int,
      z_key_dim: int,
      z_val_dim: int,
      key_aligner: Literal["simple"] = "simple",
      key_bias: bool = False,
      key_activation_fn: Optional[str] = "relu",
      val_aligner: Literal["simple"] = "simple",
      val_bias: bool = False,
      val_activation_fn: Optional[str] = "relu",
  ):
    """Initialize aligner for textual features.

    Keys are the feature names, and vals are the feature values. For categorical
    and string features, both keys and values are textual.

    Args:
      x_key_dim: Input dimensions of feature keys.
      x_val_dim: Input dimensions of feature values.
      z_key_dim: Output dimensions of feature keys.
      z_val_dim: Output dimensions of feature values.
      key_aligner: Type of key aligner to use. Currently only "simple" is
        supported.
      key_bias: Whether to add a bias to the key aligner.
      key_activation_fn: Activation function to use for key aligner.
      val_aligner: Type of value aligner to use. Currently only "simple" is
        supported.
      val_bias: Whether to add a bias to the value aligner.
      val_activation_fn: Activation function to use for value aligner.
    """
    super().__init__()

    if key_aligner == "simple":
      self.key_align = catlib.IngesTablesSimpleTextAligner(
          in_features=x_key_dim,
          out_features=z_key_dim,
          bias=key_bias,
          activation_fn=key_activation_fn,
      )

    if val_aligner == "simple":
      self.val_align = catlib.IngesTablesSimpleTextAligner(
          in_features=x_val_dim,
          out_features=z_val_dim,
          bias=val_bias,
          activation_fn=val_activation_fn,
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


class NumericAligner(nn.Module):
  """Alignment module to pass numerical features to ingestables encoder."""

  def __init__(
      self,
      x_key_dim: int,
      x_val_dim: int,
      z_key_dim: int,
      z_val_dim: int,
      key_aligner: Literal["simple"] = "simple",
      key_bias: bool = False,
      key_activation_fn: Optional[str] = "relu",
      val_aligner: Literal["simple", "periodic", "identity"] = "simple",
      val_bias: bool = False,
      val_activation_fn: Optional[str] = "relu",
      n_frequencies: Optional[int] = 48,
      frequency_init_scale: Optional[float] = 0.01,
  ):
    """Initialize aligner for numerical features.

    Keys are the feature names, and vals are the feature values. For numeric
    features, the keys are textual, while values are numerical.

    Args:
      x_key_dim: Input dimensions of feature keys.
      x_val_dim: Input dimensions of feature values.
      z_key_dim: Output dimensions of feature keys.
      z_val_dim: Output dimensions of feature values.
      key_aligner: Type of key aligner to use. Currently only "simple" is
        supported.
      key_bias: Whether to add a bias to the key aligner.
      key_activation_fn: Activation function to use for key aligner.
      val_aligner: Type of value aligner to use.
      val_bias: Whether to add a bias to the value aligner.
      val_activation_fn: Activation function to use for value aligner.
      n_frequencies: the number of frequencies for each feature. This is only
        used for periodic embeddings.
      frequency_init_scale: the initialization scale for the first linear layer.
        This is an important hyper-parameter. This is only used for periodic
        embeddings.
    """
    super().__init__()

    if key_aligner == "simple":
      self.key_align = catlib.IngesTablesSimpleTextAligner(
          in_features=x_key_dim,
          out_features=z_key_dim,
          bias=key_bias,
          activation_fn=key_activation_fn,
      )

    if val_aligner == "simple":
      self.val_align = numlib.IngesTablesSimpleNumericAligner(
          in_features=x_val_dim,
          out_features=z_val_dim,
          bias=val_bias,
          activation_fn=val_activation_fn,
      )
    elif val_aligner == "periodic":
      if x_val_dim != 1:
        raise ValueError("Periodic aligner only supports 1D features")
      self.val_align = numlib.PeriodicActEmbeddings(
          d_embedding=z_val_dim,
          activation_fn=val_activation_fn,
          bias=val_bias,
          n_frequencies=n_frequencies,
          frequency_init_scale=frequency_init_scale,
      )
    elif val_aligner == "identity":
      self.val_align = nn.Identity()
    else:
      raise ValueError(f"Unsupported val_aligner: {val_aligner}")

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
