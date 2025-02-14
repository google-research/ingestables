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

"""IngesTables kv_combiners.

How to combine key and value embeddings into one embedding.
"""

import torch
from torch import nn


class Concatenate(nn.Module):
  """Concatenate."""

  def forward(
      self,
      z_key_emb: torch.Tensor,
      z_val_emb: torch.Tensor,
  ) -> torch.Tensor:
    """Concatenate."""
    return torch.cat((z_key_emb, z_val_emb), dim=-1)


class Add(nn.Module):
  """Add."""

  def forward(
      self,
      z_key_emb: torch.Tensor,
      z_val_emb: torch.Tensor,
  ) -> torch.Tensor:
    """Add."""
    return z_key_emb + z_val_emb


class Scale(nn.Module):
  """Scale."""

  def forward(
      self,
      z_key_emb: torch.Tensor,
      z_val_emb: torch.Tensor,
  ) -> torch.Tensor:
    """Add."""
    return z_key_emb * z_val_emb
