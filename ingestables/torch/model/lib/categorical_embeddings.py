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

"""Embeddings for Categorical features."""

from typing import List, Optional

from ingestables.torch.model.lib import activations
import torch
from torch import nn
from torch.nn import parameter

__all__ = ['FTTransformerCategoricalEmbeddings', 'IngesTablesSimpleTextAligner']


class IngesTablesSimpleTextAligner(nn.Module):
  """Alignment module to pass text features to ingestables encoder."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      bias: bool = False,
      activation_fn: Optional[str] = 'relu',
  ):
    super().__init__()
    self.simple_text_aligner = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    )

    self.activation_fn = (
        activations.get_activation_fn(activation_fn)
        if activation_fn is not None
        else nn.Identity()
    )

  def forward(
      self,
      x_keys: torch.Tensor,
  ) -> torch.Tensor:
    """Embed a text cell before feeding into backbone.

    Args:
      x_keys: [..., x_key_dim] float tensor.

    Returns:
      [..., z_key_dim] float tensor.
    """
    k = self.simple_text_aligner(x_keys)
    return self.activation_fn(k)


class FTTransformerCategoricalEmbeddings(nn.Module):
  """Embeddings for categorical features for FT-Transformer.

  Learns a d-dimensional embedding for each category of each feature. For each
  feature, a separate trainable vector is added to the embedding regardless of
  a feature value. This was used in FT-Transformer.

  **Examples**

  >>> cardinalities = [3, 10]
  >>> x = torch.Tensor([
  ...     [0, 5],
  ...     [1, 7],
  ...     [0, 2],
  ...     [2, 4]
  ... ])
  >>> x.shape  # (batch_size, n_cat_features)
  torch.Size([4, 2])
  >>> m = CategoricalEmbeddings(cardinalities, d_embedding=5)
  >>> m(x).shape  # (batch_size, n_cat_features, d_embedding)
  torch.Size([4, 2, 5])

  **References**

  - Gorishniy, Yury, et al. "Revisiting deep learning models for tabular data."
  Advances in Neural Information Processing Systems 34 (2021): 18932-18943.
  -
  https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/rtdl_revisiting_models.py
  """

  def __init__(
      self,
      cardinalities: List[int],
      d_embedding: int,
      bias: bool = True,
  ) -> None:
    """Embeddings for categorical features.

    Args:
      cardinalities: the number of distinct values for each feature.
      d_embedding: the embedding size.
      bias: if `True`, for each feature, a trainable vector is added to the
        embedding regardless of a feature value. For each feature, a separate
        non-shared bias vector is allocated. In the paper, FT-Transformer uses
        `bias=True`.
    """
    super().__init__()
    if cardinalities is None:
      raise ValueError('cardinalities must not be empty')
    if any(x <= 0 for x in cardinalities):
      i, value = next((i, x) for i, x in enumerate(cardinalities) if x <= 0)
      raise ValueError(
          'cardinalities must contain only positive values,'
          f' however: cardinalities[{i}]={value}'
      )
    if d_embedding <= 0 or d_embedding is None:
      raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

    self.embeddings = nn.ModuleList(
        [nn.Embedding(x, d_embedding) for x in cardinalities]
    )
    self.bias = (
        parameter.Parameter(torch.empty(len(cardinalities), d_embedding))
        if bias
        else None
    )
    self.reset_parameters()

  def reset_parameters(self) -> None:
    d_rsqrt = self.embeddings[0].embedding_dim ** -0.5
    for m in self.embeddings:
      nn.init.uniform_(m.weight, -d_rsqrt, d_rsqrt)
    if self.bias is not None:
      nn.init.uniform_(self.bias, -d_rsqrt, d_rsqrt)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Do the forward pass."""
    if x.ndim < 2:
      raise ValueError(
          f'The input must have at least two dimensions, however: {x.ndim=}'
      )
    n_features = len(self.embeddings)
    if x.shape[-1] != n_features:
      raise ValueError(
          'The last input dimension (the number of categorical features) must'
          ' be equal to the number of cardinalities passed to the constructor.'
          f' However: {x.shape[-1]=}, len(cardinalities)={n_features}'
      )

    x = torch.stack(
        [self.embeddings[i](x[..., i]) for i in range(n_features)], dim=-2
    )
    if self.bias is not None:
      x = x + self.bias
    return x


class TransTabTextEmbeddings(nn.Module):
  """Categorical embeddings based on TransTab.

  WIP: Text is first tokenized and then embedded. The embeddings matrix is
  learned.
  """

  def __init__(
      self, cardinalities: List[int], d_embedding: int, bias: bool = True
  ) -> None:
    pass
