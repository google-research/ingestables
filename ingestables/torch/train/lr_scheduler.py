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

"""Learning rate schedulers.

Adapted from:
https://github.com/moment-timeseries-foundation-model/moment-research/blob/3ab637e413f35f2c317573c0ace280d825c558de/moment/utils/optims.py
"""

import math
from typing import List, Optional
import torch


class LinearWarmupCosineLRScheduler:
  """Learning rate scheduler with linear warmup and cosine decay.

  Attributes:
    optimizer: The optimizer to use.
    max_steps: The maximum number of steps to train for.
    warmup_steps: The number of warmup steps.
    warmup_start_lr: The starting learning rate for warmup.
    warmup_end_lr: The ending learning rate for warmup.
    peak_lr: The peak learning rate for cosine decay.
    min_lr: The minimum learning rate for cosine decay.
    simulate_lr_schedule: Whether to simulate the learning rate schedule.
    simulated_learning_rates: The simulated learning rates.
  """

  def __init__(
      self,
      optimizer: Optional[torch.optim.Optimizer],
      max_steps: int,
      warmup_steps: int = 0,
      warmup_start_lr: float = 0.0,
      warmup_end_lr: float = -1.0,
      peak_lr: float = 1e-4,
      min_lr: float = 1e-5,
      simulate_lr_schedule: bool = False,
  ):
    self.optimizer = optimizer

    self.max_steps = max_steps
    self.warmup_steps = warmup_steps
    self.warmup_start_lr = warmup_start_lr
    # If warmup_end_lr is not set, set it to peak_lr.
    self.warmup_end_lr = warmup_end_lr if warmup_end_lr >= 0 else peak_lr
    self.peak_lr = peak_lr
    self.min_lr = min_lr

    self.simulate_lr_schedule = simulate_lr_schedule
    self.simulated_learning_rates = []

    if not self.simulate_lr_schedule and self.optimizer is None:
      raise ValueError("Optimizer must be set if simulate_lr_schedule is True.")

    self._step_count = 0
    self._last_lr = []
    if self.optimizer is not None:
      self._last_lr: List[float] = [
          group["lr"] for group in self.optimizer.param_groups
      ]

  def step(self):
    """Updates the learning rate for the current step."""
    if self._step_count <= self.warmup_steps:
      self._current_lr = linear_warmup_lr_schedule(
          optimizer=self.optimizer,
          cur_step=self._step_count,
          max_step=self.warmup_steps,
          init_lr=self.warmup_start_lr,
          max_lr=self.warmup_end_lr,
      )
    else:
      self._current_lr = cosine_lr_schedule(
          optimizer=self.optimizer,
          cur_step=self._step_count,
          max_steps=self.max_steps,
          min_lr=self.min_lr,
          max_lr=self.peak_lr,
      )

    if self.simulate_lr_schedule:
      self.simulated_learning_rates.append(self._current_lr)

    if self.optimizer is not None:
      self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    self._step_count += 1

  def get_last_lr(self) -> List[float]:
    """Return last computed learning rate by current scheduler."""
    return self._last_lr

  def get_simulated_learning_rates(self) -> List[float]:
    """Return simulated learning rates."""
    return self.simulated_learning_rates


def linear_warmup_lr_schedule(
    optimizer: Optional[torch.optim.Optimizer],
    cur_step: int,
    max_step: int,
    init_lr: float,
    max_lr: float,
) -> float:
  """Linear warmup learning rate schedule.

  Args:
    optimizer: Wrapped optimizer.
    cur_step: The current step.
    max_step: The maximum number of steps to train for.
    init_lr: The starting learning rate for warmup.
    max_lr: The ending learning rate for warmup.

  Returns:
    The learning rate for the current step.
  """
  lr = min(max_lr, init_lr + (max_lr - init_lr) * cur_step / max(max_step, 1))

  if optimizer is not None:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr

  return lr


def cosine_lr_schedule(
    optimizer: Optional[torch.optim.Optimizer],
    min_lr: float,
    cur_step: int,
    max_steps: int,
    max_lr: float,
) -> float:
  """Cosine decay learning rate schedule.

  Args:
    optimizer: Wrapped optimizer.
    min_lr: The minimum learning rate for cosine decay.
    cur_step: The current step.
    max_steps: The maximum number of steps to train for.
    max_lr: The starting learning rate for cosine decay.

  Returns:
    The learning rate for the current step.
  """
  lr = min_lr + 0.5 * (max_lr - min_lr) * (
      1.0 + math.cos(math.pi * cur_step / max_steps)
  )

  if optimizer is not None:
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr

  return lr
