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

"""IngesTables model."""

import logging
from typing import Dict, List, Optional, Tuple
from etils import epath
from ingestables.torch import types
from ingestables.torch.model.backbones import t5_transformer  # pylint: disable=g-importing-member
import torch
from torch import nn
import transformers


ROOT_DIR = "~/ingestables/"


def get_params(
    model: torch.nn.Module, exclude_params: Optional[List[str]] = None
) -> List[Tuple[str, torch.tensor]]:
  """Get the parameters of a model.

  Args:
    model: The model to get the parameters of.
    exclude_params: A list of parameter names to exclude.

  Returns:
    A list of tuples, where each tuple contains the name of a parameter and its
    value.
  """
  if exclude_params is None:
    return list(model.named_parameters())
  exclude_params = set(exclude_params)

  params = []
  for name, param in model.named_parameters():
    if not any([ename in name for ename in exclude_params]):
      params.append((name, param))

  return params


class Encoder(nn.Module):
  """IngesTables encoder."""

  def __init__(
      self,
      aligners: Dict[str, nn.Module],
      special_tokens: Dict[str, nn.Module],
      kv_combiner: Dict[str, nn.Module],
      backbone: nn.Module,
  ):
    """Assembles several components into a single encoder.

    Args:
      aligners: Dict[str, Module]. Here, each key refers to distinct modalities,
        e.g., numeric, categorical, string features. Each Module takes
        `x_key_emb`, `x_val_emb`, `mask`, and `missing` Tensors, and returns
        `z_emb`. See `aligner.py`'s CatAligner and NumAligner for example
        implementations.
      special_tokens: Dict[str, Module]. Module that handles masking and
        missingness.
      kv_combiner: Dict[str, Module]. Each function combines `z_key_emb` and
        `z_val_emb` into `z_emb`. We have one unique combiner for each modality.
      backbone: Module that takes `z_emb`, and returns `z_emb`. See `backbones`
        for an example implementation.
    """
    super().__init__()
    self.aligners = nn.ModuleDict(aligners)
    self.special_tokens = nn.ModuleDict(special_tokens)
    self.kv_combiner = kv_combiner
    self.backbone = backbone

  def forward(
      self,
      inference_inputs: Dict[str, types.IngesTablesInferenceInputs],
  ) -> Dict[str, torch.Tensor]:
    """Produce embeddings for each input type.

    Args:
      inference_inputs: A dictionary of IngesTablesInferenceInputs, where each
        key corresponds to a different input type (e.g., "num", "cat", "str").

    Returns:
      Dict[str, [..., num_features, z_dim]] float tensor.
        Each str key corresponds the aligner keys. num_features corresponds to
        that of its aligner module.
    """
    z_emb_list = []
    z_emb_keys = []
    for aligner_key, aligner_fn in self.aligners.items():
      # [NOTE] We might not have numeric, categorical, or string features.
      if aligner_key not in inference_inputs:
        logging.info("No inference inputs for aligner key: %s", aligner_key)
        continue
      z_emb_keys.append(aligner_key)
      z_key_emb, z_val_emb = aligner_fn(
          x_keys=inference_inputs[aligner_key].x_keys,
          x_vals=inference_inputs[aligner_key].x_vals,
      )
      z_val_emb = self.special_tokens[aligner_key](
          z_val_emb,
          mask=inference_inputs[aligner_key].mask,
          missing=inference_inputs[aligner_key].missing,
      )
      # z_key_emb is of shape: (batch_size, num_features, z_key_dim)
      # z_val_emb is of shape: (batch_size, num_features, z_val_dim)
      z_emb = self.kv_combiner[aligner_key](z_key_emb, z_val_emb)

      # z_emb is of shape: (batch_size, num_features, z_dim)
      z_emb_list.append(z_emb)
    # Concatenate along features dimension.
    z_emb = torch.cat(z_emb_list, dim=-2)
    # z_emb is of shape: (batch_size, num_features, z_dim)
    z_emb = self.backbone(z_emb=z_emb)
    # z_emb is of shape: (batch_size, num_features, z_dim)
    # Split along features dimension.
    num_feats_per_type = [z.shape[-2] for z in z_emb_list]
    z_emb_split = torch.split(z_emb, num_feats_per_type, dim=-2)
    return {
        aligner_key: z_emb
        for aligner_key, z_emb in zip(z_emb_keys, z_emb_split)
    }


class Model(nn.Module):
  """IngesTables model."""

  def __init__(
      self,
      aligners: Dict[str, nn.Module],
      special_tokens: Dict[str, nn.Module],
      kv_combiner: Dict[str, nn.Module],
      backbone: nn.Module,
      heads: Dict[str, nn.Module],
  ):
    """Assembles several components into a single model.

    Args:
      aligners: Dict[str, Module]. Here, each key refers to distinct modalities,
        e.g., numeric, categorical, string features. Each Module takes an
        "inference_inputs" dictionary, and returns a tuple `z_key_emb,
        z_val_emb`. See `aligner.py`'s CatAligner and NumAligner for example
        implementations.
      special_tokens: Dict[str, Module] Module that handles masking and
        missingness.
      kv_combiner: Dict[str, Module]. Each function combines `z_key_emb` and
        `z_val_emb` into `z_emb`. We have one unique combiner for each modality.
        See `kv_combiner.py` for example implementations.
      backbone: Module that takes `z_emb`, and returns `z_emb`. See
        `backbone.py`'s Transformer for an example implementation.
      heads: Dict[str, Module]. Here, each key refers to distinct modalities,
        e.g., numeric, categorical, string features. Note that the set of keys
        for heads must be a subset of the set of keys in aligners. Each Module
        takes `z_emb` and other kwargs to product logits. See `head.py`'s
        Classification and Regression for example implementations.
    """
    super().__init__()
    self.encoder = Encoder(
        aligners=aligners,
        special_tokens=special_tokens,
        kv_combiner=kv_combiner,
        backbone=backbone,
    )
    self.heads = nn.ModuleDict(heads)

  def forward(
      self, inference_inputs: dict[str, types.IngesTablesInferenceInputs]
  ) -> Dict[str, torch.Tensor]:
    """Produce logits for each input type.

    Args:
      inference_inputs: A dictionary of IngesTablesInferenceInputs, where each
        key corresponds to a different input type (e.g., "numeric",
        "categorical", "string").

    Returns:
      dict[str, torch.Tensor] float Tensor. The str key corresponds to the head
        keys, the inner tensor corresponds to the output of the corresponding
        head's forward method (usually logits).
    """  # fmt: skip
    z_embs = self.encoder(inference_inputs)  # type: Dict[str, torch.Tensor]
    logits_dict = {}

    heads_keys_to_compute_logits = (
        set(self.heads.keys())
        & set(inference_inputs.keys())
        & set(z_embs.keys())
    )
    logging.info(
        "heads_keys_to_compute_logits: %s", heads_keys_to_compute_logits
    )
    for key in heads_keys_to_compute_logits:
      logits_dict[key] = self.heads[key](z_embs[key], inference_inputs[key])
    return logits_dict

  def loss(
      self,
      logits: Dict[str, torch.Tensor],
      training_inputs: Dict[str, types.IngesTablesTrainingInputs],
  ) -> Dict[str, torch.Tensor]:
    """Compute the losses using each head.

    Args:
      logits: Dict[str, float tensor]. The output of Model.forward(). The key
        corresonds to the key of the head in `self.heads.keys()`. The value is
        the output of each head's forward().
      training_inputs: The outer key corresponds to the head key, the values are
        the kwargs needed by each head to compute the loss (typically the
        labels). If a key is not provides, the loss for that head will not be
        computed.

    Returns:
      Dict mapping the head key to the outputs of head.loss().
    """
    losses_dict = {}
    heads_keys_to_compute_loss = (
        set(self.heads.keys())
        & set(logits.keys())
        & set(training_inputs.keys())
    )
    for head_key in heads_keys_to_compute_loss:
      head = self.heads[head_key]
      losses_dict[head_key] = head.loss(
          logits[head_key],
          training_inputs[head_key],
      )
    return losses_dict

  def _freeze_backbone(self, freeze: bool = True):
    """Freeze parameters of the backbone.

    Args:
      freeze: Whether to freeze the parameters of the backbone. If True, the
        parameters of the backbone are frozen. Otherwise, the parameters of the
        backbone are not frozen.
    """
    for param_name, param in self.encoder.backbone.named_parameters():
      param.requires_grad = not freeze
      logging.info(
          "Setting requires_grad of parameter %s of backbone to %s",
          param_name,
          not freeze,
      )

  def _freeze_aligners(self, freeze: bool = True):
    """Freeze parameters of aligners.

    Args:
      freeze: Whether to freeze the parameters of the aligners. If True, the
        parameters of the aligners are frozen. Otherwise, the parameters of the
        aligners are not frozen.
    """
    for aligner_name, aligner in self.encoder.aligners.items():
      for param_name, param in aligner.named_parameters():
        param.requires_grad = not freeze
        logging.info(
            "Setting requires_grad of parameter %s of %s aligner to %s",
            param_name,
            aligner_name,
            not freeze,
        )

  def _freeze_heads(
      self, freeze_head_params: bool = True, freeze_aligner_params: bool = True
  ):
    """Freeze parameters of heads.

    Args:
      freeze_head_params: Whether to freeze the parameters of the heads. If
        True, the parameters of the heads are frozen. Otherwise, the parameters
        of the heads are not frozen.
      freeze_aligner_params: Whether to freeze the parameters of the aligners.
        If True, the parameters of the aligners are frozen. Otherwise, the
        parameters of the aligners are not frozen. Remember that the
        classification head uses an aligner.
    """
    for head_name, head in self.heads.items():
      for param_name, param in head.named_parameters():
        if "aligner" in param_name:
          param.requires_grad = not freeze_aligner_params
        else:
          param.requires_grad = not freeze_head_params
          logging.info(
              "Setting requires_grad of parameter %s of %s head to %s",
              param_name,
              head_name,
              not freeze_head_params,
          )

  def _freeze_special_tokens(self, freeze: bool = True):
    """Freeze parameters of special tokens."""
    for special_token in self.encoder.special_tokens.values():
      for _, param in special_token.named_parameters():
        param.requires_grad = not freeze

  def freeze_parameters(
      self,
      freeze_backbone: bool = True,
      freeze_aligners: bool = True,
      freeze_heads: bool = False,
      freeze_special_tokens: bool = True,
  ):
    """Freeze parameters.

    If set to True, the parameters of the backbone, aligners, and heads are
    frozen. Otherwise, the parameters of the backbone, aligners, and heads are
    not frozen.

    Args:
      freeze_backbone: Whether to freeze the parameters of the backbone.
      freeze_aligners: Whether to freeze the parameters of the aligners.
      freeze_heads: Whether to freeze the parameters of the heads.
      freeze_special_tokens: Whether to freeze the parameters of the special
        tokens.
    """
    self._freeze_backbone(freeze_backbone)
    self._freeze_aligners(freeze_aligners)
    self._freeze_heads(freeze_heads, freeze_aligners)
    self._freeze_special_tokens(freeze_special_tokens)

  def load_pretrained_weights(
      self,
      ingestables_checkpoint_name_and_path: Optional[str] = None,
      pretrained_backbone_name_and_path: Optional[
          str
      ] = ROOT_DIR + "huggingface/t5-efficient-tiny",
      load_backbone_weights: bool = True,
      load_aligner_weights_if_available: bool = False,
      load_head_weights_if_available: bool = False,
  ):
    """Function to load pre-trained weights from a checkpoint.

    This implements 2 functionality:
    - Load pre-trained weights from an IngesTables checkpoint (this inclues
    weights of the backbone, heads, and aligners).
    - Load pre-trained weights of the backbone from a T5-effiecient checkpoint.

    Args:
      ingestables_checkpoint_name_and_path: The name and path of the IngesTables
        checkpoint.
      pretrained_backbone_name_and_path: The name and path of the T5-efficient
        checkpoint.
      load_backbone_weights: Whether to load the weights of the backbone.
      load_aligner_weights_if_available: Whether to load the weights of the
        aligners if available.
      load_head_weights_if_available: Whether to load the weights of the heads
        if available.
    """

    # TODO(mononito): Remove functionality to load pre-trained weights from
    # IngesTables checkpoints here. We instead load IngesTables pre-trained
    # weights when loading fiddle config.

    DO_NOT_COPY_PARAMS = ["embed_tokens", "relative_attention_bias"]  # pylint: disable=invalid-name

    if not isinstance(self.encoder.backbone, t5_transformer.T5EncoderModel):
      raise NotImplementedError(
          "Loading pre-trained weights is only implemented for T5EncoderModel"
          + " backbones."
      )
    if (
        (ingestables_checkpoint_name_and_path is not None)
        and (load_head_weights_if_available)
        and (load_aligner_weights_if_available)
    ):
      # Load pre-trained weights from an IngesTables checkpoint.
      raise NotImplementedError(
          "Loading pre-trained weights from an IngesTables checkpoint is not"
          + " implemented yet."
      )
    elif (
        pretrained_backbone_name_and_path is not None
    ) and load_backbone_weights:
      # Load pre-trained weights from a T5-efficient checkpoint.
      logging.info(
          "Loading pre-trained weights from a T5-efficient checkpoint: %s",
          pretrained_backbone_name_and_path,
      )
      pretrained_model = transformers.AutoModel.from_pretrained(
          epath.Path(pretrained_backbone_name_and_path),
          local_files_only=True,
      ).get_encoder()  # get the encoder block
      pretrained_model.to(self.encoder.backbone.device)

      pre_params = get_params(pretrained_model, DO_NOT_COPY_PARAMS)
      back_params = get_params(self.encoder.backbone, DO_NOT_COPY_PARAMS)

      if len(pre_params) != len(back_params):
        raise ValueError(
            "The number of parameters in the pretrained model does not match"
            " the number of parameters in the backbone."
        )

      for (pre_name, pre_param), (back_name, back_param) in zip(
          pre_params, back_params
      ):
        if (pre_name == back_name) and (pre_param.shape == back_param.shape):
          back_param.data.copy_(pre_param.data)
          logging.info("Loaded pre-trained weights for parameter: %s", pre_name)
        else:
          logging.warning(
              "The shape or the name of the parameter '%s' does not match the"
              " shape or the name of the parameter '%s' of the backbone.",
              pre_name,
              back_name,
          )

      # Check if weights are copied correctly
      for (pre_name, pre_param), (back_name, back_param) in zip(
          pre_params,
          back_params,
      ):
        if (pre_name == back_name) and (pre_param.shape == back_param.shape):
          assert torch.allclose(back_param.data, pre_param.data)
        else:
          raise ValueError(
              "The shape or the name of the parameter '%s' does not match the"
              " shape or the name of the parameter '%s' of the backbone."
              % (pre_name, back_name)
          )

      del pretrained_model  # release memory
