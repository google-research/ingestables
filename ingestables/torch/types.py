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

"""Type constants used throughout IngesTables."""

import dataclasses
from typing import List, Optional, Protocol
import torch


# This class is needed because the torch.dataset does not expose the
# configuration space that we want.
class Dataset:

  def __init__(self):
    pass

  def get_dataset(self):  # return a torch dataset
    pass


@dataclasses.dataclass
class Datasets:
  """A single dataset, or multiple datasets."""

  train: Dataset
  eval_on_train: Optional[Dataset] = None
  eval_on_val: Optional[Dataset] = None
  eval_on_test: Optional[Dataset] = None


@dataclasses.dataclass(kw_only=True)
class TaskInfo(Protocol):

  task_type: str
  dataset_name: str


@dataclasses.dataclass(kw_only=True)
class SupervisedTaskInfo(TaskInfo):
  target_key: str


@dataclasses.dataclass(kw_only=True)
class ClassificationTaskInfo(SupervisedTaskInfo):
  """Classification task info."""

  task_type: str = "classification"
  dataset_name: str
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
class RegressionTaskInfo(SupervisedTaskInfo):

  task_type: str = "regression"
  dataset_name: str


@dataclasses.dataclass(kw_only=True)
class UnsupervisedTaskInfo(TaskInfo):

  task_type: str = "unsupervised"
  dataset_name: str


def create_task_info(
    task_type: str,
    target_key: Optional[str] = None,
    target_classes: Optional[list[str]] = None,
    dataset_name: Optional[str] = None,
) -> TaskInfo:
  """Creates a TaskInfo object from the given task type and target info.

  Args:
    task_type: The type of task.
    target_key: The key of the target feature.
    target_classes: The classes of the target feature.
    dataset_name: The name of the dataset.

  Returns:
    A TaskInfo object.
  """
  if task_type == "classification":
    return ClassificationTaskInfo(
        target_key=target_key,
        target_classes=target_classes,
        dataset_name=dataset_name,
    )
  elif task_type == "regression":
    return RegressionTaskInfo(target_key=target_key, dataset_name=dataset_name)
  elif task_type == "unsupervised":
    return UnsupervisedTaskInfo(dataset_name=dataset_name)
  else:
    raise ValueError(f"unsupported task type: {task_type}")


@dataclasses.dataclass
class IngesTablesInferenceInputs:
  """Container for inference inputs.

  Attributes:
    x_keys: torch.Tensor of shape [..., num_features, x_key_dim] float tensor.
      Each str key corresponds the aligner keys. num_features and x_key_dim
      corresponds to that of its aligner module.
    x_vals: torch.Tensor of shape [..., num_features, x_val_dim] float tensor.
      Each str key corresponds the aligner keys. num_features and x_val_dim
      corresponds to that of its aligner module.
    missing: torch.Tensor of shape [..., num_features, 1] bool tensor. Each str
      key corresponds the aligner keys. num_features corresponds to that of its
      aligner module.
    mask: torch.Tensor of shape [..., num_features, 1] bool tensor. Each str key
      corresponds the aligner keys. num_features corresponds to that of its
      aligner module.
    x_vals_all: Optional[torch.Tensor] of shape [..., num_features, x_val_dim].
    padding: Optional[torch.Tensor] of shape [..., num_features, 1] bool tensor.
  """

  x_keys: torch.Tensor
  x_vals: torch.Tensor
  missing: torch.Tensor
  mask: torch.Tensor | None = None
  x_vals_all: torch.Tensor | None = None
  padding: torch.Tensor | None = None


@dataclasses.dataclass
class IngesTablesTrainingInputs:
  """Container for training inputs.

  Attributes:
    y_vals: torch.Tensor of shape [..., num_features, 1] int64 or float tensor.
      This is the index of the category for categorical features and the raw
      numerical value for numerical features.
    loss_weights: torch.Tensor of shape [..., num_features, 1] float tensor.
      This is multiplied element-wise to the loss for each instance and for each
      feature.
  """

  y_vals: torch.Tensor = None
  loss_weights: torch.Tensor = None


@dataclasses.dataclass
class IngesTablesEvaluationInputs:
  """Container for eval inputs.

  Attributes:
    target_index: torch.Tensor of shape [..., 1, 1] int64 tensor. This is the
      index along the features axis whose eval metrics we care about.
  """

  target_index: Optional[torch.Tensor] = None


class Metrics(Protocol):

  def metrics_dict(self) -> dict[str, float | None]:
    ...

  @property
  def head_key(self) -> str:
    ...


@dataclasses.dataclass
class ClassificationMetrics(Metrics):
  """Classification metrics.

  All metrics are computed using
  [sklearn.metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html).

  Attributes:
    accuracy: Classification accuracy score.
    log_loss: Logistic loss or cross entropy.
    f1_score: F1 score.
    auc_roc: Area under the ROC curve.
    auc_pr: Area under the precision-recall curve. This is also known as average
      precision.
  """

  accuracy: float
  log_loss: float | None = None
  f1_score: float | None = None
  auc_roc: float | None = None
  auc_pr: float | None = None

  def metrics_dict(self) -> dict[str, float | None]:
    return dataclasses.asdict(self)

  @property
  def head_key(self) -> str:
    return "cat"


@dataclasses.dataclass
class RegressionMetrics(Metrics):
  """Regression metrics.

  All metrics are computed using
  [sklearn.metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html).

  Attributes:
    mean_squared_error:
    mean_absolute_error:
    root_mean_squared_error:
    r_squared:
  """
  mean_squared_error: float
  mean_absolute_error: float | None = None
  root_mean_squared_error: float | None = None
  r_squared: float | None = None

  def metrics_dict(self) -> dict[str, float | None]:
    return dataclasses.asdict(self)

  @property
  def head_key(self) -> str:
    return "num"
