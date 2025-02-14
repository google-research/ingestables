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

# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

"""PyTorch T5 Encoder-only model without Positional Embeddings.

Source:
https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/t5/modeling_t5.py
"""

from ingestables.torch.model.lib import activations
import torch
from torch import nn
from transformers import configuration_utils
from transformers import file_utils
from transformers import modeling_utils
from transformers.utils import logging


logger = logging.get_logger(__name__)

T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/config.json",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/config.json",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/config.json",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/config.json",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/config.json",
}


class T5Config(configuration_utils.PretrainedConfig):
  r"""This is the configuration class to store the configuration of a [`T5Model`].

  It is used to instantiate a T5 model according to the specified arguments,
  defining the model architecture. Instantiating a configuration with the
  defaults will yield a similar configuration to that of the T5
  [t5-small](https://huggingface.co/t5-small) architecture.

  Configuration objects inherit from [`PretrainedConfig`] and can be used to
  control the model outputs. Read the documentation from [`PretrainedConfig`]
  for more information.
  """

  model_type = "t5"
  attribute_map = {
      "hidden_size": "d_model",
      "num_attention_heads": "num_heads",
      "num_hidden_layers": "num_layers",
  }

  def __init__(
      self,
      d_model=512,
      d_kv=64,
      d_ff=2048,
      num_layers=6,
      num_heads=8,
      dropout_rate=0.1,
      layer_norm_epsilon=1e-6,
      initializer_factor=1.0,
      feed_forward_proj="gated-gelu",
      **kwargs,
  ):
    self.d_model = d_model
    self.d_kv = d_kv
    self.d_ff = d_ff
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dropout_rate = dropout_rate
    self.layer_norm_epsilon = layer_norm_epsilon
    self.initializer_factor = initializer_factor
    self.feed_forward_proj = feed_forward_proj
    super().__init__()


T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
]


class T5LayerNorm(nn.Module):
  """T5 Layer Norm."""

  def __init__(self, hidden_size, eps=1e-6):
    """Construct a layernorm module in the T5 style No bias and no subtraction of mean."""
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states):
    # layer norm should always be calculated in float32
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(
        variance + self.variance_epsilon
    )

    # convert into half-precision if necessary
    if self.weight.dtype in [torch.float16, torch.bfloat16]:
      hidden_states = hidden_states.to(self.weight.dtype)

    return self.weight * hidden_states


class T5DenseReluDense(nn.Module):
  """Dense T5 Feed Foraward Layer."""

  def __init__(self, config):
    super().__init__()
    self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
    self.dropout = nn.Dropout(config.dropout_rate)

  def forward(self, hidden_states):
    hidden_states = self.wi(hidden_states)
    hidden_states = nn.functional.relu(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


class T5DenseGatedGeluDense(nn.Module):
  """Dense T5 Feed Foraward (Gated GeLU) Layer."""

  def __init__(self, config):
    super().__init__()
    self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
    self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
    self.dropout = nn.Dropout(config.dropout_rate)
    self.gelu_act = activations.get_activation_fn("gelu_new")

  def forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states


class T5LayerFF(nn.Module):
  """T5 Feed Forward Layer."""

  def __init__(self, config):
    super().__init__()
    if config.feed_forward_proj == "relu":
      self.DenseReluDense = T5DenseReluDense(config)  # pylint: disable=invalid-name
    elif config.feed_forward_proj == "gated-gelu":
      self.DenseReluDense = T5DenseGatedGeluDense(config)
    else:
      raise ValueError(
          f"{self.config.feed_forward_proj} is not supported. Choose between"
          " `relu` and `gated-gelu`"
      )

    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = nn.Dropout(config.dropout_rate)

  def forward(self, hidden_states):
    forwarded_states = self.layer_norm(hidden_states)
    forwarded_states = self.DenseReluDense(forwarded_states)
    hidden_states = hidden_states + self.dropout(forwarded_states)
    return hidden_states


class T5Attention(nn.Module):
  """T5 Attention Class."""

  def __init__(self, config: T5Config):
    super().__init__()
    self.is_decoder = config.is_decoder

    self.d_model = config.d_model
    self.key_value_proj_dim = config.d_kv
    self.n_heads = config.num_heads
    self.dropout = config.dropout_rate
    self.inner_dim = self.n_heads * self.key_value_proj_dim

    # Mesh TensorFlow initialization to avoid scaling before softmax
    self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
    self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    self.pruned_heads = set()

  def prune_heads(self, heads):
    if len(heads) == 0:  #  pylint: disable=g-explicit-length-test
      return
    heads, index = modeling_utils.find_pruneable_heads_and_indices(
        heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
    )
    # Prune linear layers
    self.q = modeling_utils.prune_linear_layer(self.q, index)
    self.k = modeling_utils.prune_linear_layer(self.k, index)
    self.v = modeling_utils.prune_linear_layer(self.v, index)
    self.o = modeling_utils.prune_linear_layer(self.o, index, dim=1)
    # Update hyper params
    self.n_heads = self.n_heads - len(heads)
    self.inner_dim = self.key_value_proj_dim * self.n_heads
    self.pruned_heads = self.pruned_heads.union(heads)

  def forward(
      self,
      hidden_states,
      mask=None,
      key_value_states=None,
      past_key_value=None,
      layer_head_mask=None,
      query_length=None,
      use_cache=False,
      output_attentions=False,
  ):
    """Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states)."""
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or
    # (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
      assert len(past_key_value) == 2, (
          "past_key_value should have 2 past states: keys and values. Got"
          f" { len(past_key_value)} past states"
      )
      real_seq_length += (
          past_key_value[0].shape[2] if query_length is None else query_length
      )

    def shape(states):
      """Projection."""
      return states.view(
          batch_size, -1, self.n_heads, self.key_value_proj_dim
      ).transpose(1, 2)

    def unshape(states):
      """Reshape."""
      return (
          states.transpose(1, 2)
          .contiguous()
          .view(batch_size, -1, self.inner_dim)
      )

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
      """Projects hidden states correctly to key/query states."""
      if key_value_states is None:
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(hidden_states))
      elif past_key_value is None:
        # cross-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(proj_layer(key_value_states))

      if past_key_value is not None:
        if key_value_states is None:
          # self-attn
          # (batch_size, n_heads, key_length, dim_per_head)
          hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
        else:
          # cross-attn
          hidden_states = past_key_value
      return hidden_states

    # get query states
    query_states = shape(
        self.q(hidden_states)
    )  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    # compute scores
    scores = torch.matmul(query_states, key_states.transpose(3, 2))
    # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states),
    # compatible with onnx op>9

    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
      attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(
        torch.matmul(attn_weights, value_states)
    )  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,)

    if output_attentions:
      outputs = outputs + (attn_weights,)
    return outputs


class T5LayerSelfAttention(nn.Module):
  """T5 Self Attention Layer."""

  def __init__(self, config):
    super().__init__()
    self.SelfAttention = T5Attention(config)  # pylint: disable=invalid-name
    self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
    self.dropout = nn.Dropout(config.dropout_rate)

  def forward(
      self,
      hidden_states,
      attention_mask=None,
      layer_head_mask=None,
      past_key_value=None,
      use_cache=False,
      output_attentions=False,
  ):
    normed_hidden_states = self.layer_norm(hidden_states)
    attention_output = self.SelfAttention(
        normed_hidden_states,
        mask=attention_mask,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    hidden_states = hidden_states + self.dropout(attention_output[0])
    outputs = (hidden_states,) + attention_output[
        1:
    ]  # add attentions if we output them
    return outputs


class T5Block(nn.Module):
  """T5 Block."""

  def __init__(self, config):
    super().__init__()
    self.layer = nn.ModuleList()
    self.layer.append(T5LayerSelfAttention(config))
    self.layer.append(T5LayerFF(config))

  def forward(
      self,
      hidden_states,
      attention_mask=None,
      encoder_hidden_states=None,
      encoder_attention_mask=None,
      layer_head_mask=None,
      past_key_value=None,
      use_cache=False,
      output_attentions=False,
      return_dict=True,
  ):
    self_attn_past_key_value = None

    self_attention_outputs = self.layer[0](
        hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        past_key_value=self_attn_past_key_value,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )
    hidden_states, present_key_value_state = self_attention_outputs[:2]
    attention_outputs = self_attention_outputs[
        2:
    ]  # Keep self-attention outputs and relative position weights

    # clamp inf values to enable fp16 training
    if (
        hidden_states.dtype == torch.float16
        and torch.isinf(hidden_states).any()
    ):
      clamp_value = torch.finfo(hidden_states.dtype).max - 1000
      hidden_states = torch.clamp(
          hidden_states, min=-clamp_value, max=clamp_value
      )

    # Apply Feed Forward layer
    hidden_states = self.layer[-1](hidden_states)

    # clamp inf values to enable fp16 training
    if (
        hidden_states.dtype == torch.float16
        and torch.isinf(hidden_states).any()
    ):
      clamp_value = torch.finfo(hidden_states.dtype).max - 1000
      hidden_states = torch.clamp(
          hidden_states, min=-clamp_value, max=clamp_value
      )

    outputs = (hidden_states,)

    if use_cache:
      outputs = outputs + (present_key_value_state,) + attention_outputs
    else:
      outputs = outputs + attention_outputs

    # hidden-states, present_key_value_states, (self-attention weights)
    return outputs


class T5PreTrainedModel(modeling_utils.PreTrainedModel):
  """An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models."""

  config_class = T5Config
  base_model_prefix = "transformer"

  @property
  def dummy_inputs(self):
    input_ids = torch.tensor(file_utils.DUMMY_INPUTS)
    input_mask = torch.tensor(file_utils.DUMMY_MASK)
    dummy_inputs = {
        "decoder_input_ids": input_ids,
        "input_ids": input_ids,
        "decoder_attention_mask": input_mask,
    }
    return dummy_inputs

  def _init_weights(self, module):
    """Initialize the weights."""
    factor = (
        self.config.initializer_factor
    )  # Used for testing weights initialization
    if isinstance(module, T5LayerNorm):
      module.weight.data.fill_(factor * 1.0)
    elif isinstance(module, T5DenseReluDense):
      # Mesh TensorFlow FF initialization
      # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56  # pylint: disable=line-too-long
      # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
      module.wi.weight.data.normal_(
          mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
      )
      if hasattr(module.wi, "bias") and module.wi.bias is not None:
        module.wi.bias.data.zero_()
      module.wo.weight.data.normal_(
          mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
      )
      if hasattr(module.wo, "bias") and module.wo.bias is not None:
        module.wo.bias.data.zero_()
    elif isinstance(module, T5DenseGatedGeluDense):
      module.wi_0.weight.data.normal_(
          mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
      )
      if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
        module.wi_0.bias.data.zero_()
      module.wi_1.weight.data.normal_(
          mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
      )
      if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
        module.wi_1.bias.data.zero_()
      module.wo.weight.data.normal_(
          mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
      )
      if hasattr(module.wo, "bias") and module.wo.bias is not None:
        module.wo.bias.data.zero_()
    elif isinstance(module, T5Attention):
      # Mesh TensorFlow attention initialization to avoid scaling before softmax
      # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136  # pylint: disable=line-too-long
      d_model = self.config.d_model
      key_value_proj_dim = self.config.d_kv
      n_heads = self.config.num_heads
      module.q.weight.data.normal_(
          mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5)
      )
      module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
      module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
      module.o.weight.data.normal_(
          mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5)
      )


class T5EncoderModel(T5PreTrainedModel):
  """T5 Encoder Model."""

  def __init__(self, config):
    super().__init__(config)

    self.block = nn.ModuleList(
        [T5Block(config) for _ in range(config.num_layers)]
    )
    self.final_layer_norm = T5LayerNorm(
        config.d_model, eps=config.layer_norm_epsilon
    )
    self.dropout = nn.Dropout(config.dropout_rate)

    # Initialize weights and apply final processing
    self.post_init()

  def forward(
      self,
      z_emb=None,
      attention_mask=None,
  ):
    input_shape = z_emb.size()[:-1]
    batch_size, seq_length = input_shape

    if attention_mask is None:
      attention_mask = torch.ones(batch_size, seq_length).to(z_emb.device)
    extended_attention_mask = self.get_extended_attention_mask(
        attention_mask, input_shape, z_emb.device
    )

    # Prepare head mask if needed
    hidden_states = self.dropout(z_emb)

    for layer_module in self.block:
      layer_outputs = layer_module(
          hidden_states,
          attention_mask=extended_attention_mask,
          encoder_hidden_states=None,
          encoder_attention_mask=None,
          layer_head_mask=None,
          past_key_value=None,
          use_cache=False,
          output_attentions=False,
      )

      # layer_outputs is a tuple with:
      # hidden-states, key-value-states, (self-attention weights), ...
      hidden_states = layer_outputs[0]

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    return hidden_states
