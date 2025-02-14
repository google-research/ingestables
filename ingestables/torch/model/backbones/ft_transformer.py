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

"""FT-Transformer Encoderbackbone.

Source:
https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/rtdl_revisiting_models.py
"""

import typing
from typing import Literal, Optional, cast

from ingestables.torch.model.lib import utils
import ingestables.torch.model.lib.deep as deep_lib
import torch
from torch import nn
import torch.nn.functional as F


_TransformerFFNActivation = Literal['ReLU', 'ReGLU']
_LINFORMER_KV_COMPRESSION_SHARING = Literal['headwise', 'key-value']  # pylint: disable=invalid-name


class _ReGLU(nn.Module):

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
      raise ValueError(
          'For the ReGLU activation, the last input dimension'
          f' must be a multiple of 2, however: {x.shape[-1]=}'
      )
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


class FTTransformerBackbone(nn.Module):
  """The backbone of FT-Transformer.

  The differences with Transformer from the paper
  ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) are as
  follows:

  - the so called "PreNorm" variation is used
    (`norm_first=True` in terms of `torch.nn.TransformerEncoderLayer`)
  - the very first normalization is skipped. This is **CRUCIAL** for
    FT-Transformer in the PreNorm configuration.

  **Examples**

  >>> batch_size = 2
  >>> n_tokens = 3
  >>> d_block = 16
  >>> x = torch.randn(batch_size, n_tokens, d_block)
  >>> m = FTTransformerBackbone(
  ...     n_blocks=2,
  ...     d_block=d_block,
  ...     attention_n_heads=8,
  ...     attention_dropout=0.2,
  ...     ffn_d_hidden=None,
  ...     ffn_d_hidden_multiplier=2.0,
  ...     ffn_dropout=0.1,
  ...     residual_dropout=0.0,
  ... )
  >>> m(x).shape
  torch.Size([2, 3, 16])
  """

  def __init__(
      self,
      *,
      n_blocks: int,
      d_block: int,
      attention_n_heads: int,
      attention_dropout: float,
      ffn_d_hidden: Optional[int] = None,
      ffn_d_hidden_multiplier: Optional[float],
      ffn_dropout: float,
      # NOTE[DIFF]
      # In the paper, FT-Transformer uses the ReGLU activation.
      # Here, to illustrate the difference, ReLU activation is also supported
      # (in particular, see the docstring).
      ffn_activation: _TransformerFFNActivation = 'ReGLU',
      residual_dropout: float,
      n_tokens: Optional[int] = None,
      linformer_kv_compression_ratio: Optional[float] = None,
      linformer_kv_compression_sharing: Optional[
          _LINFORMER_KV_COMPRESSION_SHARING
      ] = None,
  ):
    """The backbone of FT-Transformer.

    Args:
      n_blocks: the number of blocks.
      d_block: the block width (or, equivalently, the embedding size of each
        feature). Must be a multiple of `attention_n_heads`.
      attention_n_heads: the number of attention heads in `MultiheadAttention`.
      attention_dropout: the dropout rate in `MultiheadAttention`. Usually,
        positive values work better, even if the number of features is low.
      ffn_d_hidden: the hidden representation size after the activation in the
        feed-forward blocks (or, equivalently, the *input* size of the *second*
        linear layer in the feed-forward blocks). If ``ffn_use_reglu`` is
        `True`, then the *output* size of the *first* linear layer will be set
        to ``2 * ffn_d_hidden``.
      ffn_d_hidden_multiplier: the alternative way to set `ffn_d_hidden` as
        `int(d_block * ffn_d_hidden_multiplier)`.
      ffn_dropout: the dropout rate for the hidden representation in the
        feed-forward blocks.
      ffn_activation: the activation used in the FFN blocks. To maintain
        (almost) the same number of parameters between different activations:
        <ffn_d_hidden_multiplier for ReGLU> = <2 / 3 * ffn_d_hidden_multiplier
        for ReLU> or <ffn_d_hidden_multiplier for ReLU> = <3 / 2 *
        ffn_d_hidden_multiplier for ReGLU>
      residual_dropout: the dropout rate for all residual branches.
      n_tokens: the argument for `MultiheadAttention`.
      linformer_kv_compression_ratio: the argument for `MultiheadAttention`.
      linformer_kv_compression_sharing: the argument for `MultiheadAttention`.
    """  # noqa: E501
    if ffn_activation not in typing.get_args(_TransformerFFNActivation):
      raise ValueError(
          'ffn_activation must be one of'
          f' {typing.get_args(_TransformerFFNActivation)}.'
          f' However: {ffn_activation=}'
      )
    if ffn_d_hidden is None:
      if ffn_d_hidden_multiplier is None:
        raise ValueError(
            'If ffn_d_hidden is None,'
            ' then ffn_d_hidden_multiplier must not be None'
        )
      ffn_d_hidden = int(d_block * cast(float, ffn_d_hidden_multiplier))
    else:
      if ffn_d_hidden_multiplier is not None:
        raise ValueError(
            'If ffn_d_hidden is not None,'
            ' then ffn_d_hidden_multiplier must be None'
        )

    super().__init__()
    ffn_use_reglu = ffn_activation == 'ReGLU'
    self.blocks = nn.ModuleList([
        nn.ModuleDict({
            # >>> attention
            'attention': deep_lib.MultiheadAttention(
                d_embedding=d_block,
                n_heads=attention_n_heads,
                dropout=attention_dropout,
                n_tokens=n_tokens,
                linformer_kv_compression_ratio=linformer_kv_compression_ratio,
                linformer_kv_compression_sharing=linformer_kv_compression_sharing,
            ),
            'attention_residual_dropout': nn.Dropout(residual_dropout),
            # >>> feed-forward
            'ffn_normalization': nn.LayerNorm(d_block),
            'ffn': utils._named_sequential(
                (
                    'linear1',
                    # ReGLU divides dimension by 2,
                    # so multiplying by 2 to compensate for this.
                    nn.Linear(
                        d_block, ffn_d_hidden * (2 if ffn_use_reglu else 1)
                    ),
                ),
                ('activation', _ReGLU() if ffn_use_reglu else nn.ReLU()),
                ('dropout', nn.Dropout(ffn_dropout)),
                ('linear2', nn.Linear(ffn_d_hidden, d_block)),
            ),
            'ffn_residual_dropout': nn.Dropout(residual_dropout),
            # >>> output (for hook-based introspection)
            'output': nn.Identity(),
            # >>> the very first normalization
            **(
                {}
                if layer_idx == 0
                else {'attention_normalization': nn.LayerNorm(d_block)}
            ),
        })
        for layer_idx in range(n_blocks)
    ])
    # if d_out is None:
    #   self.output = None
    # else:
    #   self.output = utils._named_sequential(
    #       ('normalization', nn.LayerNorm(d_block)),
    #       ('activation', nn.ReLU()),
    #       ('linear', nn.Linear(d_block, d_out)),
    #   )

  def forward(self, z_emb: torch.Tensor) -> torch.Tensor:
    """Do the forward pass."""
    if z_emb.ndim != 3:
      raise ValueError(
          f'The input must have exactly three dimension, however: {z_emb.ndim=}'
      )

    n_blocks = len(self.blocks)
    for i_block, block in enumerate(self.blocks):
      block = cast(nn.ModuleDict, block)

      z_emb_identity = z_emb
      if 'attention_normalization' in block:
        z_emb = block['attention_normalization'](z_emb)
      z_emb = block['attention'](
          z_emb[:, :1] if i_block + 1 == n_blocks else z_emb, z_emb
      )
      z_emb = block['attention_residual_dropout'](z_emb)
      z_emb = z_emb_identity + z_emb

      z_emb_identity = z_emb
      z_emb = block['ffn_normalization'](z_emb)
      z_emb = block['ffn'](z_emb)
      z_emb = block['ffn_residual_dropout'](z_emb)
      z_emb = z_emb_identity + z_emb

      z_emb = block['output'](z_emb)

    # z_emb = z_emb[:, 0]  # The representation of [CLS]-token.

    # if self.output is not None:
    #   z_emb = self.output(z_emb)
    return z_emb
