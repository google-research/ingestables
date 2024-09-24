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

import dataclasses
from typing import Any, Callable

import torch
from torch import nn


def apply_mask_and_missing(
    z_val_emb: torch.Tensor,
    *,
    mask: torch.Tensor,
    mask_emb: torch.Tensor,
    missing: torch.Tensor,
    missing_emb: torch.Tensor,
) -> torch.Tensor:
  """Apply mask and missing to aligned embedding."""
  missing_emb = missing_emb.expand(*(z_val_emb.shape[:-1] + (-1,)))
  z_val_emb = torch.where(missing, z_val_emb, missing_emb)
  mask_emb = mask_emb.expand(*(z_val_emb.shape[:-1] + (-1,)))
  z_val_emb = torch.where(mask, z_val_emb, mask_emb)
  return z_val_emb


class Encoder(nn.Module):
  """IngesTables encoder."""

  def __init__(
      self,
      aligners: dict[str, nn.Module],
      kv_combiner: nn.Module,
      backbone: nn.Module,
      mask_emb: nn.Parameter,
      missing_emb: nn.Parameter,
  ):
    """Assembles several components into a single encoder.

    Args:
      aligners: dict[str, Module]. Here, each key refers to distinct modalities,
        e.g, numeric, categorical, string features. Each Module takes
        `x_key_emb`, `x_val_emb`, `mask`, and `missing` Tensors, and returns
        `z_emb`. See `aligner.py`'s CatAligner and NumAligner for example
        implementations.
      kv_combiner: Module that combines `z_key_emb` and `z_val_emb` into
        `z_emb`.
      backbone: Module that takes `z_emb`, and returns `z_emb`. See
        `backbone.py`'s Transformer for an example implementation.
      mask_emb: Wherever `mask` is 0, replace z_val_emb with this.
      missing_emb: Wherever `missing` is 0, replace z_val_emb with this.
    """
    super().__init__()
    self.aligners = nn.ModuleDict(aligners)
    self.kv_combiner = kv_combiner
    self.backbone = backbone
    # Putting them in a ParameterDict is necessary to register them as part of
    # this module.
    self.special_tokens = nn.ParameterDict({
        "mask": mask_emb,
        "missing": missing_emb,
    })

  def _apply_mask_and_missing(
      self,
      z_val_emb: torch.Tensor,
      *,
      mask: torch.Tensor,
      missing: torch.Tensor,
  ) -> torch.Tensor:
    """Apply mask and missing to aligned embedding."""
    return apply_mask_and_missing(
        z_val_emb,
        mask=mask,
        mask_emb=self.special_tokens["mask"],
        missing=missing,
        missing_emb=self.special_tokens["missing"],
    )

  def forward(
      self,
      inference_inputs: dict[str, dict[str, torch.Tensor]],
  ) -> dict[str, torch.Tensor]:
    """Produce embeddings for each input type.

    Args:
      inference_inputs: dict containing:
        "x_keys": dict[str, [..., num_features, x_key_dim]] float tensor.
          Each str key corresponds the aligner keys. num_features and x_key_dim
          corresponds to that of its aligner module.
        "x_vals": dict[str, [..., num_features, x_val_dim]] float tensor.
          Each str key corresponds the aligner keys. num_features and x_val_dim
          corresponds to that of its aligner module.
        "mask": dict[str, [..., num_features, 1]] bool tensor.
          Each str key corresponds the aligner keys. num_features corresponds to
          that of its aligner module.
        "missing": dict[str, [..., num_features, 1]] bool tensor. Each str key
          corresponds the aligner keys. num_features corresponds to that of its
          aligner module.
        (and other items needed by the aligner module)

    Returns:
      dict[str, [..., num_features, z_dim]] float tensor.
        Each str key corresponds the aligner keys. num_features corresponds to
        that of its aligner module.
    """  # fmt: skip
    z_emb_list = []
    for aligner_key, aligner_fn in self.aligners.items():
      z_key_emb, z_val_emb = aligner_fn(
          x_keys=inference_inputs[aligner_key]["x_keys"],
          x_vals=inference_inputs[aligner_key]["x_vals"],
      )
      z_val_emb = self._apply_mask_and_missing(
          z_val_emb,
          mask=inference_inputs[aligner_key]["mask"],
          missing=inference_inputs[aligner_key]["missing"],
      )
      z_emb = self.kv_combiner(z_key_emb, z_val_emb)
      z_emb_list.append(z_emb)
    # Concatenate along features dimension.
    z_emb = torch.cat(z_emb_list, dim=-2)
    z_emb = self.backbone(z_emb)
    # Split along features dimension.
    num_feats_per_type = [z.shape[-2] for z in z_emb_list]
    z_emb_split = torch.split(z_emb, num_feats_per_type, dim=-2)
    return {
        aligner_key: z_emb
        for aligner_key, z_emb in zip(self.aligners.keys(), z_emb_split)
    }


@dataclasses.dataclass
class EncoderConfig:
  """A container for all configuration options of Encoder.

  See `encoder_from_config` for how it is used.
  """
  # Map input types to aligner configs (see aligner.py).
  aligners: dict[str, Any]
  # The output dim of all aligners' z_val_emb.
  z_val_dim: int
  # Name of the KV combiner.
  kv_combiner_type: str
  # Name of the backbone.
  backbone_type: str
  # The backbone config (see backbone.py).
  backbone: Any


def encoder_from_config(
    aligner_cls_dict: dict[str, Callable[..., nn.Module]],
    kv_combiner_cls_dict: dict[str, Callable[..., nn.Module]],
    backbone_cls_dict: dict[str, Callable[..., nn.Module]],
    config: EncoderConfig,
) -> Encoder:
  """Construct an ingestables.Encoder instance from a config."""
  aligners = {
      aligner_key: aligner_cls_dict[aligner_key](aligner_cfg)
      for aligner_key, aligner_cfg in config.aligners.items()
  }
  kv_combiner = kv_combiner_cls_dict[config.kv_combiner_type]()
  backbone = backbone_cls_dict[config.backbone_type](config.backbone)
  mask_emb = torch.rand(config.z_val_dim, dtype=torch.float32)
  missing_emb = torch.rand(config.z_val_dim, dtype=torch.float32)
  return Encoder(
      aligners=aligners,
      kv_combiner=kv_combiner,
      backbone=backbone,
      mask_emb=mask_emb,
      missing_emb=missing_emb,
  )


class Model(nn.Module):
  """IngesTables model."""

  def __init__(
      self,
      aligners: dict[str, nn.Module],
      kv_combiner: nn.Module,
      backbone: nn.Module,
      heads: dict[str, nn.Module],
      mask_emb: nn.Parameter,
      missing_emb: nn.Parameter,
  ):
    """Assembles several components into a single model.

    Args:
      aligners: dict[str, Module]. Here, each key refers to distinct modalities,
        e.g, numeric, categorical, string features. Each Module takes
        an "inference_inputs" dictionary, and returns a tuple
        `z_key_emb, z_val_emb`. See `aligner.py`'s CatAligner and
        NumAligner for example implementations.
      kv_combiner: Module that combines `z_key_emb, z_val_emb` into `z_emb`.
        See `kv_combiner.py` for example implementations.
      backbone: Module that takes `z_emb`, and returns `z_emb`. See
        `backbone.py`'s Transformer for an example implementation.
      heads: dict[str, Module]. Here, each key refers to distinct modalities,
        e.g, numeric, categorical, string features. Note that the set of keys
        for heads must be a subset of the set of keys in aligners. Each Module
        takes `z_emb` and other kwargs to product logits. See `head.py`'s
        Classification and Regression for example implementations.
      mask_emb: Wherever `mask` is 0, replace z_val_emb with this.
      missing_emb: Wherever `missing` is 0, replace z_val_emb with this.
    """
    super().__init__()
    self.encoder = Encoder(
        aligners=aligners,
        kv_combiner=kv_combiner,
        backbone=backbone,
        mask_emb=mask_emb,
        missing_emb=missing_emb,
    )
    self.heads = nn.ModuleDict(heads)

  def forward(
      self, inference_inputs: dict[str, dict[str, torch.Tensor]]
  ) -> dict[str, torch.Tensor]:
    """Produce logits for each input type.

    Args:
      inference_inputs: dict containing:
        "x_keys": dict[str, [..., num_features,  x_key_dim]] float tensor.
          Each str key corresponds the aligner keys. num_features and x_key_dim
          corresponds to that of its aligner module.
        "x_vals": dict[str, [..., num_features, x_val_dim]] float tensor.
          Each str key corresponds the aligner keys. num_features and x_val_dim
          corresponds to that of its aligner module.
        "mask": dict[str, [..., num_features, 1]] bool tensor.
          Each str key corresponds the aligner keys. num_features corresponds to
          that of its aligner module.
        "missing": dict[str, [..., num_features, 1]] bool tensor.
          Each str key corresponds the aligner keys. num_features corresponds to
          that of its aligner module.
        (and other items needed by the aligner module)

    Returns:
      dict[str, torch.Tensor] float Tensor. The str key corresponds to the head
        keys, the inner tensor corresponds to the output of the corresponding
        head's forward method (usually logits).
    """  # fmt: skip
    z_embs = self.encoder(inference_inputs)  # type: dict[str, torch.Tensor]
    logits_dict = {}
    for key, z_emb in z_embs.items():
      if key in self.heads.keys():
        logits_dict[key] = self.heads[key](z_emb, inference_inputs[key])
    return logits_dict

  def loss(
      self,
      logits: dict[str, torch.Tensor],
      training_inputs: dict[str, dict[str, torch.Tensor]],
  ) -> dict[str, torch.Tensor]:
    """Compute the losses using each head.

    Args:
      logits: dict[str, float tensor]. The output of Model.forward(). The key
        corresonds to the key of the head in `self.heads.keys()`. The value is
        the output of each head's forward().
      training_inputs: The outer key corresponds to the head key, the values are
        the kwargs needed by each head to compute the loss (typically the
        labels). If a key is not provides, the loss for that head will not be
        computed.

    Returns:
      dict mapping the head key to the outputs of head.loss().
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


@dataclasses.dataclass
class ModelConfig:
  """A container for all configuration options of Model.

  See `model_from_config` for how it is used.
  """
  # Map input types to aligner configs (see aligner.py).
  aligners: dict[str, Any]
  # The output dim of all aligners' z_val_emb.
  z_val_dim: int
  # Name of the KV combiner.
  kv_combiner_type: str
  # Name of the backbone.
  backbone_type: str
  # The backbone config (see backbone.py).
  backbone: Any
  # Map head types to head configs (see head.py).
  heads: dict[str, Any]


def model_from_config(
    aligner_cls_dict: dict[str, Callable[..., nn.Module]],
    kv_combiner_cls_dict: dict[str, Callable[..., nn.Module]],
    backbone_cls_dict: dict[str, Callable[..., nn.Module]],
    head_cls_dict: dict[str, Callable[..., nn.Module]],
    config: ModelConfig,
) -> Model:
  """Construct an ingestables.Model instance from a config."""
  aligners = {
      aligner_key: aligner_cls_dict[aligner_key](aligner_cfg)
      for aligner_key, aligner_cfg in config.aligners.items()
  }
  kv_combiner = kv_combiner_cls_dict[config.kv_combiner_type]()
  backbone = backbone_cls_dict[config.backbone_type](config.backbone)
  heads = {
      head_key: head_cls_dict[head_key](
          head_cfg,
          aligner=aligners[head_key],
          kv_combiner=kv_combiner,
      )
      for head_key, head_cfg in config.heads.items()
  }
  mask_emb = torch.rand(config.z_val_dim, dtype=torch.float32)
  missing_emb = torch.rand(config.z_val_dim, dtype=torch.float32)
  return Model(
      aligners=aligners,
      kv_combiner=kv_combiner,
      backbone=backbone,
      heads=heads,
      mask_emb=mask_emb,
      missing_emb=missing_emb,
  )
