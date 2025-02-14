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

"""TransTab Gated Transformer."""

from typing import Optional

from ingestables.torch.model.lib import activations
import torch
from torch import nn
import torch.nn.functional as F


class TransTabTransformerLayer(nn.Module):
  """TransTab Transformer layer."""

  __constants__ = ['batch_first', 'norm_first']

  def __init__(
      self,
      d_model,
      nhead,
      dim_feedforward=2048,
      dropout=0.1,
      activation=F.relu,
      layer_norm_eps=1e-5,
      batch_first=True,
      norm_first=False,
      device=None,
      dtype=None,
      use_layer_norm=True,
  ) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.self_attn = nn.MultiheadAttention(
        d_model, nhead, batch_first=batch_first, **factory_kwargs
    )
    # Implementation of Feedforward model
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    # Implementation of gates
    self.gate_linear = nn.Linear(d_model, 1, bias=False)
    self.gate_act = nn.Sigmoid()

    self.norm_first = norm_first
    self.use_layer_norm = use_layer_norm

    if self.use_layer_norm:
      self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
      self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    # Legacy string support for activation function.
    if isinstance(activation, str):
      self.activation = activations.get_activation_fn(activation)
    else:
      self.activation = activation

  # self-attention block
  def _sa_block(
      self,
      x: torch.Tensor,
      attn_mask: Optional[torch.Tensor],
      key_padding_mask: Optional[torch.Tensor],
  ) -> torch.Tensor:
    key_padding_mask = (
        ~key_padding_mask.bool() if key_padding_mask is not None else None
    )
    x = self.self_attn(
        x,
        x,
        x,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )[0]
    return self.dropout1(x)

  # feed forward block
  def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
    g = self.gate_act(self.gate_linear(x))
    h = self.linear1(x)
    h = h * g  # add gate
    h = self.linear2(self.dropout(self.activation(h)))
    return self.dropout2(h)

  def __setstate__(self, state):
    if 'activation' not in state:
      state['activation'] = F.relu
    super().__setstate__(state)

  def forward(
      self,
      src,
      src_mask=None,
      src_key_padding_mask=None,
      **kwargs,
  ) -> torch.Tensor:
    r"""Pass the input through the encoder layer.

    Args:
        src: the sequence to the encoder layer (required).
        src_mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).
        **kwargs: additional arguments

    Returns:
        The output of this encoder layer.
    """
    # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
    x = src
    if self.use_layer_norm:
      if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
      else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))

    else:  # do not use layer norm
      x = x + self._sa_block(x, src_mask, src_key_padding_mask)
      x = x + self._ff_block(x)
    return x


class TransTabEncoder(nn.Module):
  """TransTab encoder."""

  def __init__(
      self,
      hidden_dim=128,
      num_layer=2,
      num_attention_head=2,
      hidden_dropout_prob=0,
      ffn_dim=256,
      activation='relu',
  ):
    super().__init__()
    self.transformer_encoder = nn.ModuleList([
        TransTabTransformerLayer(
            d_model=hidden_dim,
            nhead=num_attention_head,
            dropout=hidden_dropout_prob,
            dim_feedforward=ffn_dim,
            batch_first=True,
            layer_norm_eps=1e-5,
            norm_first=False,
            use_layer_norm=True,
            activation=activation,
        )
    ])
    if num_layer > 1:
      encoder_layer = TransTabTransformerLayer(
          d_model=hidden_dim,
          nhead=num_attention_head,
          dropout=hidden_dropout_prob,
          dim_feedforward=ffn_dim,
          batch_first=True,
          layer_norm_eps=1e-5,
          norm_first=False,
          use_layer_norm=True,
          activation=activation,
      )
      stacked_transformer = nn.TransformerEncoder(
          encoder_layer, num_layers=num_layer - 1
      )
      self.transformer_encoder.append(stacked_transformer)

  def forward(
      self,
      embedding: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      **kwargs,
  ) -> torch.Tensor:
    """Forward pass.

    Args:
      embedding: batch_size, num_token, hidden_dim
      attention_mask: batch_size, num_token
      **kwargs: additional arguments

    Returns:
      embedding: batch_size, num_token, hidden_dim
    """
    outputs = embedding
    for _, mod in enumerate(self.transformer_encoder):
      outputs = mod(outputs, src_key_padding_mask=attention_mask)
    return outputs
