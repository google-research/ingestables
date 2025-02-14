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

"""Ingestables built-in backbone."""

import torch
from torch import nn


class SwiGLU(nn.Module):
  """Implements the SwiGLU activation."""

  def __init__(self):
    super().__init__()
    self.silu = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    x, gate = torch.split(x, half_dim, dim=-1)
    return torch.multiply(self.silu(gate), x)


class MLP(nn.Module):
  """Multilayer perceptron."""

  def __init__(self, z_dim: int, dropout_mlp: float):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(
            in_features=z_dim,
            out_features=z_dim * 4,
            bias=False,
        ),
        SwiGLU(),
        nn.Linear(
            in_features=z_dim * 2,
            out_features=z_dim,
            bias=False,
        ),
        nn.Dropout(dropout_mlp),
    )

  def forward(self, z_emb: torch.Tensor) -> torch.Tensor:
    return self.layers(z_emb)


class TransformerLayer(nn.Module):
  """Transformer layer."""

  def __init__(
      self,
      z_dim: int,
      num_heads: int,
      dropout_attn: float,
      dropout_mlp: float,
  ):
    super().__init__()
    self.ln = nn.LayerNorm(
        normalized_shape=z_dim,
        bias=False,
    )
    self.attn = nn.MultiheadAttention(
        embed_dim=z_dim,
        num_heads=num_heads,
        dropout=dropout_attn,
        bias=False,
        batch_first=True,
    )
    self.mlp = MLP(z_dim=z_dim, dropout_mlp=dropout_mlp)

  def forward(self, z_emb: torch.Tensor) -> torch.Tensor:
    """Transformer layer."""
    post_ln_z_emb = self.ln(z_emb)
    attn_outs, _ = self.attn(
        query=z_emb,
        key=z_emb,
        value=z_emb,
    )
    mlp_outs = self.mlp(post_ln_z_emb)
    return z_emb + attn_outs + mlp_outs


class Transformer(nn.Module):
  """Transformer."""

  def __init__(self, layers: list[TransformerLayer]):
    super().__init__()
    self.layers = nn.Sequential(*layers)

  def forward(self, z_emb: torch.Tensor) -> torch.Tensor:
    """Transformer."""
    return self.layers(z_emb)
