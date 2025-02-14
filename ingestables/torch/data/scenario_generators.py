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

"""Scenario Generator: Splitting, and Sampling datasets to orchestrate real-world transfer learning scenarios."""

from typing import Dict, Literal, Optional, Tuple
from ingestables.torch import types
import numpy as np
import pandas as pd
import sklearn.model_selection as sklearn_model_selection
import sklearn.preprocessing as sklearn_preprocessing


class Splitter:
  """Class to split a dataset into train, validation, and test splits.

  Splits a single dataset systematically or randomly into a train, validation,
  and test splits.
  """

  def __init__(
      self,
      split_type: Literal["random"] = "random",
      train_ratio: float = 0.70,
      val_ratio: float = 0.10,
      test_ratio: float = 0.20,
      random_state: int = -1,
  ):
    self.split_type = split_type
    self.train_ratio = train_ratio
    self.val_ratio = val_ratio
    self.test_ratio = test_ratio
    self.random_state = random_state

  def __repr__(self):
    return (
        f"Splitter(split_type={self.split_type},"
        + f" train_ratio={self.train_ratio},"
        + f" val_ratio={self.val_ratio},"
        + f" test_ratio={self.test_ratio},"
        + f" random_state={self.random_state})"
    )

  def __call__(
      self,
      data: pd.DataFrame,
      task_info: types.SupervisedTaskInfo,
  ) -> Tuple[Dict[str, pd.DataFrame], types.TaskInfo]:
    """Split a dataset randomly into a train, validation, and test splits."""

    if self.split_type == "random":
      return randomly_split_dataset(
          data=data,
          task_info=task_info,
          train_ratio=self.train_ratio,
          val_ratio=self.val_ratio,
          test_ratio=self.test_ratio,
          random_state=self.random_state,
      )
    else:
      raise ValueError(f"Unsupported split_type: {self.split_type}")


class Sampler:
  """Class to sample a split dataset for k-shot training and evaluation."""

  def __init__(
      self,
      sampling_type: Literal["k-shot", "full", "random"] = "full",
      k: int | float = 32,
      random_state: int = -1,
  ):
    """Initializes a Sampler.

    Args:
      sampling_type: The type of sampling to perform. "k-shot" means sampling k
        examples for each k-shot. "full" means no sampling. "random" means
        random sampling without replacement.
      k: The number of examples to sample in case of k-shot sampling. In case of
        random sampling, this is the percentage of data to sample, must lie
        between 0 and 1.
      random_state: The random state to use for sampling.
    """
    self.sampling_type = sampling_type
    self.k = int(k) if sampling_type == "k-shot" else k
    self.random_state = random_state

    if self.sampling_type == "random" and not 0 <= self.k <= 1:
      raise ValueError(
          "In case of random sampling, k must be between 0 and 1, but got"
          f" {self.k}."
      )

  def __repr__(self):
    return (
        f"Sampler(sampling_type={self.sampling_type},"
        + f" k={self.k},"
        + f" random_state={self.random_state})"
    )

  def __call__(
      self,
      data: Dict[str, pd.DataFrame],
      task_info: types.SupervisedTaskInfo,
  ) -> Tuple[Dict[str, pd.DataFrame], types.TaskInfo]:
    """Sample a split dataset for k-shot training and evaluation."""
    if self.sampling_type == "full":
      return data, task_info
    elif self.sampling_type in ["k-shot", "random"]:
      return randomly_sample_k_shots(
          data=data,
          task_info=task_info,
          k=self.k,
          random_state=self.random_state,
      )
    else:
      raise ValueError(f"Unsupported sampling_type: {self.sampling_type}")


# --------------------------------------------------------------------------
# Dataset Splitters
# Functions to split a dataset into train, validation, and test splits.
# --------------------------------------------------------------------------


def randomly_split_dataset(
    data: pd.DataFrame,
    task_info: types.SupervisedTaskInfo,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = -1,
) -> Tuple[Dict[str, pd.DataFrame], types.TaskInfo]:
  """Split a dataset randomly into a train, validation, and test splits.

  Args:
    data: pd.Dataframe
    task_info: Task information
    train_ratio: Ratio of data to be used for training.
    val_ratio: Ratio of data to be used for validation.
    test_ratio: Ratio of data to be used for testing.
    random_state: int

  Returns:
    Train, validation and test data and targets, and task information
  """
  # TODO(mononito): Add ability to split on groups
  # First split data into train + val and test splits
  task = task_info.task_type
  target = data[task_info.target_key]
  idxs = np.arange(len(data))
  stratify = target if task == "classification" else None

  assert train_ratio + val_ratio + test_ratio == 1
  assert min(train_ratio, val_ratio, test_ratio) > 0

  train_val_idxs, test_idxs = sklearn_model_selection.train_test_split(
      idxs,
      shuffle=True,
      random_state=random_state,
      stratify=stratify,
      test_size=test_ratio,
  )
  train_idxs, val_idxs = sklearn_model_selection.train_test_split(
      train_val_idxs, random_state=random_state, test_size=val_ratio
  )

  train_data = data.iloc[train_idxs, :]
  val_data = data.iloc[val_idxs, :]
  test_data = data.iloc[test_idxs, :]

  n_features = data.shape[1]
  assert train_data.shape[1] == n_features
  assert val_data.shape[1] == n_features
  assert test_data.shape[1] == n_features

  assert len(train_data) + len(val_data) + len(test_data) == len(data)
  assert np.abs((len(train_data) / len(data)) - train_ratio) < 0.1
  assert np.abs((len(val_data) / len(data)) - val_ratio) < 0.1
  assert np.abs((len(test_data) / len(data)) - test_ratio) < 0.1

  return (
      {
          "train": train_data,
          "val": val_data,
          "test": test_data,
      },
      task_info,
  )


# --------------------------------------------------------------------------
# Dataset Samplers
# Functions to sample datasets.
# --------------------------------------------------------------------------
def randomly_sample_k_shots(
    data: Dict[str, pd.DataFrame],
    task_info: types.SupervisedTaskInfo,
    k: Optional[Literal["full"] | int | float] = 32,
    random_state: int = -1,
):
  """Sample a sample a dataset for k-shot training and evaluation.

  This function randomly samples k examples from the training split and returns
  a new dataset with the sampled examples in the training split and the original
  validation and test splits.

  Args:
    data: A dictionary of pandas DataFrames, where the keys are the split names
      (e.g., "train", "val", "test").
    task_info: Task information, including the target key and task type.
    k: The number of examples to sample in case of k-shot sampling. In case of
      random sampling, this is the percentage of data to sample, must lie
      between 0 and 1.
    random_state: Random state for reproducibility.

  Returns:
    Train, validation and test data and targets, and task information
  """
  if k == "full":
    return data, task_info

  assert "train" in data, "Dataset does not have a `train` split."

  idxs = np.arange(len(data["train"]))

  if task_info.task_type == "classification":
    y = data["train"][task_info.target_key]
    y = sklearn_preprocessing.LabelEncoder().fit_transform(y)

  else:
    y = None

  train_idxs, _ = sklearn_model_selection.train_test_split(
      idxs,
      shuffle=True,
      random_state=random_state,
      stratify=y,
      train_size=k,
  )

  return (
      {
          "train": data["train"].iloc[train_idxs, :],
          "val": data["val"],
          "test": data["test"],
      },
      task_info,
  )
