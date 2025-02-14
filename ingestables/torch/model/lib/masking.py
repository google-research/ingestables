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

from typing import Tuple

from absl import logging
from ingestables.torch import types
import torch


class MaskingStrategy:
  """Masking strategy."""

  def __init__(
      self,
      target_masking_prob: float = 1.0,
      default_masking_prob: float = 0.0,
      random_seed: int = 42,
  ):
    """Make add mask function."""
    self.target_masking_prob = target_masking_prob
    self.default_masking_prob = default_masking_prob
    self.random_seed = random_seed

    self.random_gen = torch.Generator()
    self.random_gen.manual_seed(self.random_seed)

  def __repr__(self):
    return (
        f"MaskingStrategy(target_masking_prob={self.target_masking_prob},"
        f" default_masking_prob={self.default_masking_prob},"
        f" random_seed={self.random_seed})"
    )

  def generate_mask(
      self,
      target_available: bool,
      inference_inputs: types.IngesTablesInferenceInputs,
      feature_type: str,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a mask."""

    # [NOTE]: mask is of shape: (batch_size, n_features, 1)
    # loss_weights is of shape: (batch_size, n_features, 1)
    # mask is True (1) when the feature is masked, and False (0) otherwise.
    # loss_weights is 1.0 when the feature is masked, and 0.0 otherwise,
    # because we want to ignore unmasked features during loss computation.

    batch_size, n_features, _ = inference_inputs.x_keys.shape
    device = inference_inputs.x_keys.device

    if feature_type == "str":  # Do not mask string features.
      mask = torch.zeros((batch_size, n_features, 1)).bool()
      loss_weights = torch.zeros_like(mask).float()
    else:
      # TODO(mononito): This does not mask exactly the target_masking_prob
      # because the random numbers are not (always) uniform.
      rand_numbers = torch.rand(
          (batch_size, n_features, 1),
          generator=self.random_gen,
      )
      mask = rand_numbers <= self.default_masking_prob

      if target_available:
        mask[:, :1] = rand_numbers[:, :1] <= self.target_masking_prob

      loss_weights = mask.float()

    self._check_mask_and_loss_weights(
        mask, loss_weights, target_available, feature_type
    )

    return mask.bool().to(device), loss_weights.float().to(device)

  def _check_mask_and_loss_weights(
      self,
      mask: torch.Tensor,
      loss_weights: torch.Tensor,
      target_available: bool,
      feature_type: str,
  ):
    """Checks mask and loss_weights."""
    if mask.shape != loss_weights.shape:
      raise ValueError("The shape of the mask and loss_weights do not match.")

    if feature_type == "str":
      if mask.float().mean() != 0.0:
        raise ValueError("The mask for string features should be all 0.")
      if loss_weights.float().mean() != 0.0:
        raise ValueError("The loss weights for string features should be 0.")
      return

    if target_available:
      if not torch.allclose(
          mask[:, 0].float().mean(),
          torch.tensor(self.target_masking_prob),
          atol=0.2,
      ):
        logging.warn(
            "The mean of the target mask (%f) is not"
            + " equal to the target masking probability"
            + " (%f).",
            mask[:, 0].float().mean(),
            self.target_masking_prob,
        )
      if mask.shape[1] > 1 and not torch.allclose(
          mask[:, 1:].float().mean(),
          torch.tensor(self.default_masking_prob),
          atol=0.2,
      ):
        logging.warn(
            "The mean of the mask (%f) is not equal"
            + " to the default masking probability"
            + " (%f).",
            mask[:, 1:].float().mean(),
            self.default_masking_prob,
        )
    else:
      if not torch.allclose(
          mask.float().mean(), torch.tensor(self.default_masking_prob), atol=0.2
      ):
        logging.warn(
            "The mean of the mask (%f) is not equal to the"
            " default masking probability (%f)."
            "mask.shape=%s target_available=%s",
            mask.float().mean(),
            self.default_masking_prob,
            mask.shape,
            target_available,
        )
