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

"""Masking functions."""

from typing import Dict, Tuple
import ml_collections
import torch


def get_masking_config() -> ml_collections.ConfigDict:
  """Masking config."""
  config = ml_collections.ConfigDict()
  # Probability with which the target values are masked.
  config.target_masking_prob = 1.0
  # Probability with which non-target values are masked.
  config.default_masking_prob = 0.0
  config.random_seed = 42

  # TODO(scottyak): Maybe add info about feature types which shouldn't be masked
  # For example, string features should not be masked

  return config


class MaskingStrategy:
  """Masking strategy."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
  ):
    """Make add mask function."""
    self.config = config
    self.random_gen = torch.Generator()
    self.random_gen.manual_seed(config.random_seed)

    self.target_masking_prob = config.target_masking_prob
    self.default_masking_prob = config.default_masking_prob

  def __repr__(self):
    return f"MaskingStrategy(config={self.config}"

  def add_mask(
      self,
      inputs: Tuple[
          Dict[str, Dict[str, torch.Tensor]],
          Dict[str, Dict[str, torch.Tensor]],
          Dict[str, Dict[str, torch.Tensor]],
      ],
  ) -> Tuple[
      Dict[str, Dict[str, torch.Tensor]],
      Dict[str, Dict[str, torch.Tensor]],
      Dict[str, Dict[str, torch.Tensor]],
  ]:
    """Adds mask."""
    inference_inputs, training_inputs, eval_inputs = inputs

    # Infer if input is batched and the batch size
    batched_input = False
    batch_size = 1
    for v in inference_inputs.values():
      if v["x_vals"].ndim > 2:
        batch_size = v["x_vals"].shape[0]
        batched_input = True
      break

    for feat_val in inference_inputs.values():
      n_feats = (
          feat_val["x_vals"].shape[1]
          if batched_input
          else feat_val["x_vals"].shape[0]
      )
      feat_val["mask"] = (
          torch.rand(
              size=(batch_size, n_feats),
              generator=self.random_gen,
          )
          < self.default_masking_prob
      ).to(torch.bool)

    # Now find the target and mask it with target probability.
    # Since we mask all features, overwrite the mask of the target feature
    target_feat_type = list(eval_inputs.keys())[0]
    target_index = eval_inputs[target_feat_type]["target_index"]

    inference_inputs[target_feat_type]["mask"][..., target_index] = (
        torch.rand(
            size=(batch_size, 1),
            generator=self.random_gen,
        )
        < self.target_masking_prob
    ).to(torch.bool)

    return inference_inputs, training_inputs, eval_inputs
