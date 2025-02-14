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

"""Utilities."""

from collections.abc import Callable
import dataclasses
import random
from typing import Any, Dict, List, Optional

from etils import epath
from etils import etree
from ingestables.torch import types
from ingestables.torch.data import encoders
import numpy as np
import torch
from torch.optim import optimizer


OptimizerBinderFn = Callable[[optimizer.ParamsT], optimizer.Optimizer]
LearningRateSchedulerBinderFn = Callable[
    [optimizer.Optimizer], torch.optim.lr_scheduler.LRScheduler
]


initial_states = {}


def seed_everything(seed: int | None = None):
  """Seed everything."""
  if seed is None:
    return

  # TODO(joetoth): Find the op that is not deterministic.
  #   The following error is thrown if CUBLAS_WORKSPACE_CONFIG is not set.
  #
  # RuntimeError: Deterministic behavior was enabled with either
  # `torch.use_deterministic_algorithms(True)` or
  # `at::Context::setDeterministicAlgorithms(true)`, but this operation is
  # not deterministic because it uses CuBLAS and you have CUDA >= 10.2.
  # To enable deterministic behavior in this case, you must set an environment
  # variable before running your PyTorch application:
  # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more
  # information, go to
  # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility set a
  # debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit
  # overall performance) or :4096:8 (will increase library footprint in GPU
  # memory by approximately 24MiB).
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  if not initial_states:
    random.seed(seed)
    initial_states["python"] = random.getstate()
    np.random.seed(seed)
    initial_states["numpy"] = np.random.get_state(legacy=False)
    torch.manual_seed(seed)
    initial_states["torch"] = torch.random.get_rng_state()
    torch.cuda.manual_seed_all(seed)
    initial_states["torch_cuda"] = torch.cuda.get_rng_state_all()
  random.setstate(initial_states["python"])
  np.random.set_state(initial_states["numpy"])
  torch.random.set_rng_state(initial_states["torch"])
  torch.cuda.set_rng_state_all(initial_states["torch_cuda"])


def seed_worker(worker_id: int):
  del worker_id  # unused
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
  """Flattens a nested dictionary."""
  flattened = {}
  for k, v in d.items():
    if isinstance(v, dict):
      flattened.update(flatten_dict(v, prefix=k + "."))
    else:
      flattened[prefix + k] = v
  return flattened


def concatenate_metrics_kwargs(
    metrics_dict: dict[str, Any],
) -> dict[str, Any]:
  return etree.map(
      torch.concat,
      metrics_dict,
      is_leaf=lambda x: isinstance(x, list),
  )


def save_model(
    step_count, model, model_optimizer, scaler, output_path: epath.Path
):
  """Save model."""
  # torch.save() requires existing output_dir.
  output_path.mkdir(parents=True, exist_ok=True)
  with (output_path / f"model_{step_count}.pt").open("wb") as f:
    torch.save(model.state_dict(), f)
  with (output_path / f"checkpoint_{step_count}.tar").open("wb") as f:
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#saving-resuming
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        f,
    )


def training_inputs_collate_fn(
    batch: List[types.IngesTablesTrainingInputs],
) -> types.IngesTablesTrainingInputs:
  """Collates a batch of IngesTablesTrainingInputs."""
  return types.IngesTablesTrainingInputs(
      y_vals=torch.cat(
          [training_input.y_vals for training_input in batch], dim=0
      ),
      loss_weights=torch.cat(
          [training_input.loss_weights for training_input in batch], dim=0
      ),
  )


def preprocessed_inputs_collate_fn(batch: List[encoders.PreprocessedInputs]):
  """Collates a batch of PreprocessedInputs.

  Args:
    batch: The batch of PreprocessedInputs to collate.

  Returns:
    A collated PreprocessedInputs.
  """
  if batch[0].encoded_numeric is None:  # No numeric features
    encoded_numeric = None
  else:
    encoded_numeric = torch.stack([i.encoded_numeric for i in batch])

  if batch[0].encoded_categorical is None:  # No categorical features
    encoded_categorical = None
  else:
    encoded_categorical = torch.stack([i.encoded_categorical for i in batch])

  if batch[0].encoded_string is None:  # No string features
    encoded_string = None
  else:
    encoded_string = torch.stack([i.encoded_string for i in batch])

  if batch[0].encoded_targets is None:  # No targets
    encoded_targets = None
  else:
    encoded_targets = torch.stack([i.encoded_targets for i in batch])

  if batch[0].raw_numeric is None:  # No targets
    raw_numeric = None
  else:
    raw_numeric = torch.stack([i.raw_numeric for i in batch])

  if batch[0].encoded_categorical_ordinal is None:
    # When no categorical ordinal features are present
    encoded_categorical_ordinal = None
  else:
    encoded_categorical_ordinal = torch.stack(
        [i.encoded_categorical_ordinal for i in batch]
    )

  return encoders.PreprocessedInputs(
      encoded_numeric=encoded_numeric,
      encoded_categorical=encoded_categorical,
      encoded_string=encoded_string,
      encoded_targets=encoded_targets,
      raw_numeric=raw_numeric,
      encoded_categorical_ordinal=encoded_categorical_ordinal,
      encoded_feature_names=batch[0].encoded_feature_names,
      feature_type_dict=batch[0].feature_type_dict,
      categorical_value_embeddings=batch[0].categorical_value_embeddings,
      categorical_value_padding=batch[0].categorical_value_padding,
  )


def move_to_device(
    inference_inputs: Optional[
        Dict[str, types.IngesTablesInferenceInputs]
    ] = None,
    training_inputs: Optional[
        Dict[str, types.IngesTablesTrainingInputs]
    ] = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Optional[dict[str, types.IngesTablesInferenceInputs]],
    Optional[dict[str, types.IngesTablesTrainingInputs]],
]:
  """Move tensors to device."""
  if inference_inputs is not None:
    for key, inference_input in inference_inputs.items():
      if inference_input is None:
        inference_inputs[key] = None
      else:
        inference_inputs[key] = types.IngesTablesInferenceInputs(
            **etree.map(
                lambda x: x.to(device) if x is not None else None,
                dataclasses.asdict(inference_input),
            )
        )
  if training_inputs is not None:
    for key, training_input in training_inputs.items():
      if training_input is None:
        training_inputs[key] = None
      else:
        training_inputs[key] = types.IngesTablesTrainingInputs(
            **etree.map(
                lambda x: x.to(device) if x is not None else None,
                dataclasses.asdict(training_input),
            )
        )

  return inference_inputs, training_inputs


def assert_equals(actual, expected, message=None):
  """Raises AssertionError if two values are not equal. Includes the values.

  Args:
    actual: The actual value.
    expected: The expected value.
    message: An optional message to include in the error message.
  """
  if expected != actual:
    error_message = f"Assertion failed: Expected {expected} but got {actual}"
    if message:
      error_message += f": {message}"
    raise AssertionError(error_message)
