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

"""Type constants used throughout ingestables."""

import dataclasses
from typing import List, Optional, Protocol

import torch


@dataclasses.dataclass
class Datasets:
  """A single dataset, or multiple datasets."""

  train: torch.Tensor
  eval_on_train: Optional[torch.Tensor] = None
  eval_on_val: Optional[torch.Tensor] = None
  eval_on_test: Optional[torch.Tensor] = None


@dataclasses.dataclass(kw_only=True)
class TaskInfo(Protocol):

  task_type: str


@dataclasses.dataclass(kw_only=True)
class ClassificationTaskInfo(TaskInfo):
  """Classification task info."""

  task_type: str = "classification"
  target_key: str
  target_classes: List[str]

  def __post_init__(self):
    if len(self.target_classes) < 2:
      raise ValueError(
          "classification task must have at least 2 target classes; "
          f"got {len(self.target_classes)}"
      )
    if len(self.target_classes) > 2:
      raise ValueError(
          "multiclass classification not supported yet, must have exactly 2 "
          f"target classes; got {len(self.target_classes)}"
      )


@dataclasses.dataclass(kw_only=True)
class RegressionTaskInfo(TaskInfo):

  task_type: str = "regression"
  target_key: str


@dataclasses.dataclass(kw_only=True)
class UnsupervisedTaskInfo(TaskInfo):

  task_type: str = "unsupervised"
