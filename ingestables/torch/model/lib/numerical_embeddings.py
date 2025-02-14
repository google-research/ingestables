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

"""Embeddings for continuous / Numerical features.

FTTransformerLinearEmbeddings, FTTransformerLinearActEmbeddings and
PeriodicActEmbeddings adapted from
https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
"""

import collections
import math
from typing import Optional

from ingestables.torch.model.lib import activations
import torch
from torch import nn
from torch.nn import parameter


__all__ = [
    'IngesTablesSimpleNumericAligner',
    'FTTransformerLinearEmbeddings',
    'FTTransformerLinearActEmbeddings',
    'PeriodicActEmbeddings',
    'ContextualEmbeddings',
]


class IngesTablesSimpleNumericAligner(nn.Module):
  """Alignment module to pass numerical features to ingestables encoder."""

  def __init__(
      self,
      in_features: int,
      out_features: int,
      bias: bool = False,
      activation_fn: Optional[str] = 'relu',
  ):
    super().__init__()
    self.simple_numeric_aligner = nn.Linear(
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
    """Forward pass.

    Args:
      x_keys: Numerical feature keys, (*, n_features).

    Returns:
      Aligned numerical feature embeddings, (*, n_features, out_features).
    """
    return self.activation_fn(self.simple_numeric_aligner(x_keys))


class FTTransformerLinearEmbeddings(nn.Module):
  """Linear embeddings for continuous features, used in FT-Transformer.

  **Shape**

  - Input: `(*, n_features)`
  - Output: `(*, n_features, d_embedding)`

  **Examples**

  >>> batch_size = 2
  >>> n_cont_features = 3
  >>> x = torch.randn(batch_size, n_cont_features)
  >>> d_embedding = 4
  >>> m = LinearEmbeddings(n_cont_features, d_embedding)
  >>> m(x).shape
  torch.Size([2, 3, 4])
  """

  def __init__(self, n_features: int, d_embedding: int = 64):
    """Linear embeddings for continuous features.

    Args:
      n_features: the number of continous features.
      d_embedding: the embedding size.
    """
    if n_features <= 0:
      raise ValueError(f'n_features must be positive, however: {n_features=}')
    if d_embedding <= 0:
      raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

    super().__init__()
    self.weight = parameter.Parameter(torch.empty(n_features, d_embedding))
    self.bias = parameter.Parameter(torch.empty(n_features, d_embedding))
    self.reset_parameters()

  def reset_parameters(self):
    d_rqsrt = self.weight.shape[1] ** -0.5
    nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
    nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    _check_input_shape(x, self.weight.shape[0])
    return torch.addcmul(self.bias, self.weight, x[..., None])


class FTTransformerLinearActEmbeddings(nn.Sequential):
  """Linear embeddings with activations for continuous features.

  **Shape**

  - Input: `(*, n_features)`
  - Output: `(*, n_features, d_embedding)`

  **Examples**

  >>> batch_size = 2
  >>> n_cont_features = 3
  >>> x = torch.randn(batch_size, n_cont_features)
  >>>
  >>> # By default, d_embedding=32.
  >>> m = LinearActEmbeddings(n_cont_features)
  >>> m(x).shape
  torch.Size([2, 3, 32])
  """

  def __init__(
      self, n_features: int, d_embedding: int = 64, activation_fn: str = 'relu'
  ):
    super().__init__(
        collections.OrderedDict([
            (
                'linear',
                FTTransformerLinearEmbeddings(n_features, d_embedding),
            ),
            ('activation', activations.get_activation_fn(activation_fn)),
        ])
    )


class _Periodic(nn.Module):
  """Implementation of periodic embeddings for continuous features."""

  def __init__(self, k: int, sigma: float):
    if sigma <= 0.0:
      raise ValueError(f'sigma must be positive, however: {sigma=}')

    super().__init__()
    self._sigma = sigma
    self.weight = parameter.Parameter(torch.empty(k))
    self.reset_parameters()

  def reset_parameters(self):
    # Here, extreme values (~0.3% probability) are avoided just in case.
    bound = self._sigma * 3
    nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = 2 * math.pi * self.weight * x[..., None]
    x = torch.cat([torch.cos(x), torch.sin(x)], -1)
    return x


class PeriodicActEmbeddings(nn.Module):
  """Periodic embeddings for continuous features.

  The last linear layer is shared between all features.

  [DIFF] The periodic embeddings are shared between all features.

  **Shape**

  - Input: `(*, n_features)`
  - Output: `(*, n_features, d_embedding)`

  **Examples**

  >>> batch_size = 2
  >>> n_cont_features = 3
  >>> x = torch.randn(batch_size, n_cont_features)
  >>>
  >>> # PLR(lite) embeddings.
  >>> m = PeriodicActEmbeddings()
  >>> m(x).shape
  torch.Size([2, 3, 24])
  """

  def __init__(
      self,
      d_embedding: int = 64,
      activation_fn: str = 'relu',
      *,
      bias: bool = False,
      n_frequencies: int = 48,
      frequency_init_scale: float = 0.01,
  ):
    """Periodic embeddings with activations for continuous features.

    Args:
      d_embedding: the embedding size.
      activation_fn: the activation function. ReLU by default.
      bias: whether to use bias in the linear layer.
      n_frequencies: the number of frequencies for each feature.
      frequency_init_scale: the initialization scale for the first linear layer
        (denoted as "sigma" in Section 3.3 in the paper). **This is an important
        hyperparameter**.
    """
    super().__init__()
    self.periodic = _Periodic(n_frequencies, frequency_init_scale)
    self.linear = nn.Linear(2 * n_frequencies, d_embedding, bias=bias)
    self.activation = (
        nn.Identity()
        if activation_fn is None
        else activations.get_activation_fn(activation_fn)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Do the forward pass."""
    x = x.squeeze(-1)
    x = self.periodic(x)
    x = self.linear(x)
    if self.activation is not None:
      x = self.activation(x)
    return x


def _check_input_shape(x: torch.Tensor, expected_n_features: int) -> None:
  if x.ndim < 1:
    raise ValueError(
        f'The input must have at least one dimension, however: {x.ndim=}'
    )
  if x.shape[-1] != expected_n_features:
    raise ValueError(
        'The last dimension of the input was expected to be'
        f' {expected_n_features}, however, {x.shape[-1]=}'
    )


class ContextualEmbeddings(nn.Module):
  """Numerical feature embeddings conditioned on feature name."""

  def __init__(self, d_embedding: int = 64):
    super().__init__()
    self.d_embedding = d_embedding
    self.bias = nn.Parameter(torch.Tensor(1, 1, d_embedding))
    nn.init.uniform_(
        self.bias, a=-1 / math.sqrt(d_embedding), b=1 / math.sqrt(d_embedding)
    )

  def forward(
      self, feat_key_embds: torch.Tensor, x: torch.Tensor
  ) -> torch.Tensor:
    """Do the forward pass.

    Args:
      feat_key_embds: Numerical feature Key Embeddings (n_features, # tokens,
        d_embedding)
      x: numerical features, (*, n_features)

    Returns:
      Embeddings of numerical features.
    """
    assert feat_key_embds.shape[0] == x.shape[-1]
    assert self.bias.shape[-1] == feat_key_embds.shape[-1] == self.d_embedding

    feat_key_embds = feat_key_embds.unsqueeze(0).expand((x.shape[0], -1, -1))
    feat_val_embds = feat_key_embds * x.unsqueeze(-1).float() + self.bias
    return feat_val_embds
