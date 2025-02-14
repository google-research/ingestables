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

"""Top level pipeline component to manage config, data, trainers, etc.

Intended to provide functionality such as:
- Runtime setup that is common to all experiments, that takes place
  after configuration.
- Data loading.
- Initializing trainers.
- Experiment generation (inside a work unit).
  This can be helpful to run more than one extremtely short lived sklearn
  experiment per work unit XManager limit is 2k parallel work units IIRC so
  running 100s of experiments, per work unit, is faster.
- Monitoring.
"""

from absl import logging
from ingestables.torch import utils
from ingestables.torch.train import train
from ingestables.torch.train import train_deep
from ingestables.torch.train import train_sklearn
import torch
from torch._inductor.async_compile import AsyncCompile  # pylint: disable=g-importing-member

TrainerType = train.Trainer | train_deep.Trainer | train_sklearn.Trainer


def init_device(is_gpu_required: bool) -> str:
  """Prints device information and checks if a GPU is required.

  If a GPU is required, but no GPU is found, exits the program returning 1.

  Args:
    is_gpu_required: Whether a GPU is required.

  Returns:
    "cuda" if a GPU is found, "cpu" otherwise.
  """
  try:
    logging.info("is_gpu_required: %s", is_gpu_required)
    logging.info("torch.__version__: %s", torch.__version__)
    logging.info("torch.cuda.device_count(): %s", torch.cuda.device_count())
    logging.info("torch.cuda.current_device(): %s", torch.cuda.current_device())
    logging.info(
        "torch.cuda.get_device_name(0): %s", torch.cuda.get_device_name(0)
    )
    logging.info("torch.cuda.is_available(): %s", torch.cuda.is_available())
    if torch.cuda.is_available():
      return "cuda"
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(e)
  if is_gpu_required:
    logging.error("GPU required, but no GPU found.")
    exit(1)
  logging.error("Falling back to CPU.")
  return "cpu"


class TablularFoundationLab:
  """Tablular Foundation Lab."""

  def __init__(
      self,
      trainer: TrainerType,
      is_gpu_required: bool = False,
      seed: int = -1,
  ):
    self._trainer = trainer
    self._is_gpu_required = is_gpu_required
    self._seed = seed

  def __repr__(self):
    return (
        f"TabularFoundationLab(trainer={self._trainer},"
        + f" is_gpu_required={self._is_gpu_required}, seed={self._seed})"
    )

  def run(self):
    utils.seed_everything(self._seed)

    # Only initialize GPU for non-sklearn models.
    if not isinstance(self._trainer, train_sklearn.Trainer):
      init_device(self._is_gpu_required)
      AsyncCompile.warm_pool()

    self._trainer.run()

  @property
  def trainer(self) -> TrainerType:
    return self._trainer
