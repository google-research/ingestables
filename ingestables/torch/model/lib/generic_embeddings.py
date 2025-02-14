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

"""Generic embeddings."""

import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import parameter


def apply_mask_and_missing(
    z_val_emb: torch.Tensor,
    *,
    mask: torch.Tensor,
    mask_emb: torch.Tensor,
    missing: torch.Tensor,
    missing_emb: torch.Tensor,
) -> torch.Tensor:
  """Apply mask and missing to aligned embedding.

  Args:
    z_val_emb: The aligned embedding.
    mask: The mask tensor.
    mask_emb: The mask embedding tensor.
    missing: The missing tensor.
    missing_emb: The missing embedding tensor.

  Returns:
    The aligned embedding with mask and missing applied.
  """
  missing_emb = missing_emb.expand(*(z_val_emb.shape[:-1] + (-1,)))
  z_val_emb = torch.where(missing, missing_emb, z_val_emb)
  mask_emb = mask_emb.expand(*(z_val_emb.shape[:-1] + (-1,)))
  z_val_emb = torch.where(mask, mask_emb, z_val_emb)
  return z_val_emb


class IngesTablesSpecialTokens(nn.Module):
  """Container for special tokens."""

  def __init__(self, z_val_dim: int):
    """Initialize special tokens for the IngesTables model.

    Args:
      z_val_dim: The dimension of the value embeddings.
    """
    super().__init__()
    self.special_tokens = nn.ParameterDict({
        'mask': nn.Parameter(torch.rand(z_val_dim, dtype=torch.float32)),
        'missing': nn.Parameter(torch.rand(z_val_dim, dtype=torch.float32)),
    })
    self._reset_parameters()
    # TODO(mononito): CLS and Separator tokens (for multi-row inputs) can be
    # added here.

  def _reset_parameters(self):
    for _, special_token in self.special_tokens.items():
      d_rsqrt = special_token.shape[-1] ** -0.5
      nn.init.uniform_(special_token, -d_rsqrt, d_rsqrt)

  def forward(
      self,
      z_val_emb: torch.Tensor,
      mask: torch.Tensor,
      missing: torch.Tensor,
  ) -> torch.Tensor:
    """Apply mask and missing to z_emb."""
    return apply_mask_and_missing(
        z_val_emb,
        mask=mask,
        mask_emb=self.special_tokens['mask'],
        missing=missing,
        missing_emb=self.special_tokens['missing'],
    )


# TODO(mononito): Keep one implementation of the CLSEmbedding.
class CLSEmbedding(nn.Module):
  """CLS embedding used in FT-Transformer."""

  def __init__(self, d_embedding: int) -> None:
    super().__init__()
    self.weight = parameter.Parameter(torch.empty(d_embedding))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    d_rsqrt = self.weight.shape[-1] ** -0.5
    nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

  def forward(self, batch_dims: Tuple[int, ...]) -> torch.Tensor:
    if not batch_dims:
      raise ValueError('The input must be non-empty')

    return self.weight.expand(*batch_dims, 1, -1)

# NOTE: The TransTab CLSToken implementation is equivalent to the previous
# implementation of the CLSEmbedding. The only difference is the output, and
# handling of the attention mask.


class TransTabCLSToken(nn.Module):
  """add a learnable cls token embedding at the end of each sequence."""

  def __init__(self, hidden_dim) -> None:
    super().__init__()
    self.weight = nn.Parameter(torch.Tensor(hidden_dim))
    nn.init.uniform_(
        self.weight, a=-1 / math.sqrt(hidden_dim), b=1 / math.sqrt(hidden_dim)
    )
    self.hidden_dim = hidden_dim

  def expand(self, *leading_dimensions):
    new_dims = (1,) * (len(leading_dimensions) - 1)
    return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

  def forward(self, embedding, attention_mask=None, **kwargs) -> torch.Tensor:
    embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1)
    outputs = {'embedding': embedding}
    if attention_mask is not None:
      attention_mask = torch.cat(
          [
              torch.ones(attention_mask.shape[0], 1).to(attention_mask.device),
              attention_mask,
          ],
          1,
      )
    outputs['attention_mask'] = attention_mask
    return outputs
