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

"""Data pipeline.

Raw data (CSV)
 -> split (train/val/test)
 -> sample (optional)
 -> preprocess
 -> encode
 -> dataloaders
"""

import dataclasses
from typing import Tuple, cast

from absl import logging
from ingestables.torch import types
from ingestables.torch.data import data_loaders
from ingestables.torch.data import encoders
from ingestables.torch.data import preprocessors
from ingestables.torch.data import scenario_generators
from torch.utils import data


@dataclasses.dataclass
class PipelineModule:
  """Holds the components that make up part of the data pipeline."""

  benchmark_name: str
  dataset_name: str
  splitter: scenario_generators.Splitter
  sampler: scenario_generators.Sampler
  preprocessor: preprocessors.Preprocessor
  encoder: encoders.Encoder


class IndexedDataset(data.Dataset):
  """A Dataset that provides data by indexing into a preprocessed dataset."""

  def __init__(self, inputs: encoders.PreprocessedInputs):
    """Initializes the IndexedDataset.

    Args:
      inputs: The preprocessed data to index into.
    """
    super().__init__()
    self.inputs = inputs

  def __getitem__(self, idx):
    """Returns the data at the given index.

    Args:
      idx: The index of the data to return.

    Returns:
      The data at the given index.
    """

    encoded_numeric = (
        self.inputs.encoded_numeric[idx]
        if self.inputs.encoded_numeric is not None
        else None
    )

    raw_numeric = (
        self.inputs.raw_numeric[idx]
        if self.inputs.raw_numeric is not None
        else None
    )

    encoded_categorical = (
        self.inputs.encoded_categorical[idx]
        if self.inputs.encoded_categorical is not None
        else None
    )

    encoded_categorical_ordinal = (
        self.inputs.encoded_categorical_ordinal[idx]
        if self.inputs.encoded_categorical_ordinal is not None
        else None
    )

    encoded_string = (
        self.inputs.encoded_string[idx]
        if self.inputs.encoded_string is not None
        else None
    )
    encoded_targets = (
        self.inputs.encoded_targets[idx]
        if self.inputs.encoded_targets is not None
        else None
    )

    categorical_value_embeddings = (
        self.inputs.categorical_value_embeddings[0:]
        if self.inputs.categorical_value_embeddings is not None
        else None
    )
    categorical_value_padding = (
        self.inputs.categorical_value_padding[0:]
        if self.inputs.categorical_value_padding is not None
        else None
    )

    return encoders.PreprocessedInputs(
        encoded_numeric=encoded_numeric,
        encoded_categorical=encoded_categorical,
        encoded_string=encoded_string,
        encoded_targets=encoded_targets,
        encoded_feature_names=self.inputs.encoded_feature_names,
        feature_type_dict=self.inputs.feature_type_dict,
        raw_numeric=raw_numeric,
        categorical_value_embeddings=categorical_value_embeddings,
        categorical_value_padding=categorical_value_padding,
        encoded_categorical_ordinal=encoded_categorical_ordinal,
    )

  def __len__(self):
    """Returns the number of data points in the dataset.

    Returns:
      The number of data points in the dataset.
    """
    return len(self.inputs.encoded_targets)


@dataclasses.dataclass
class DatasetGroup:
  """A group of datasets for train, val, and test."""

  train: IndexedDataset
  val: IndexedDataset
  test: IndexedDataset


class Pipeline:
  """A pipeline for loading, preprocessing, and encoding data."""

  def __init__(
      self,
      pipeline_modules: list[PipelineModule],
  ):
    """Initializes the Pipeline.

    Args:
      pipeline_modules: A list of PipelineModules to include in the pipeline.
    """
    self._modules = {module.dataset_name: module for module in pipeline_modules}
    self._dataset_groups = {}
    self._task_infos = {}

    logging.info("Building dataset groups and task infos...")
    for module in self._modules.values():
      dataset_group, task_info = self._build_dataset_group_and_task_info(module)
      self._dataset_groups[module.dataset_name] = dataset_group
      self._task_infos[module.dataset_name] = task_info

  def __repr__(self):
    return (
        f"Pipeline(pipeline_modules={self._modules})"
    )

  def _build_dataset_group_and_task_info(
      self, module: PipelineModule
  ) -> Tuple[DatasetGroup, types.SupervisedTaskInfo]:
    """Builds a DatasetGroup for the given module.

    Args:
      module: The PipelineModule to build the DatasetGroup for.

    Returns:
      A DatasetGroup containing the train, val, and test datasets.
      SupervisedTaskInfo for the dataset.
    """
    df, task_info = data_loaders.load_dataset_from_benchmark(
        module.benchmark_name, module.dataset_name
    )

    # TODO(joetoth): Add support for unsupervised tasks. Splitters and samplers
    # don't work for unsupervised tasks. Refactor to support unsupervised tasks.
    task_info = cast(types.SupervisedTaskInfo, task_info)
    split_data, _ = module.splitter(df, task_info=task_info)
    sampled_data, _ = module.sampler(split_data, task_info=task_info)
    preprocessed_data = module.preprocessor(sampled_data)
    encoded_inputs = module.encoder(preprocessed_data, task_info)

    # TODO(joetoth): Shuffle here?
    return (
        DatasetGroup(
            train=IndexedDataset(encoded_inputs["train"]),
            val=IndexedDataset(encoded_inputs["val"]),
            test=IndexedDataset(encoded_inputs["test"]),
        ),
        task_info,
    )

  # TODO(joetoth): Currently the pipeline assumes a single in-memory dataset
  # that is split into train/val/test.  In order to support separate datasets
  # for train/val/test, the pipeline needs to be updated to take in separate
  # datasets.

  ##############################################################################
  # Currently the data is retrieved via indexing. The step number and batch size
  # determines the index range for the dataset. This is not ideal for
  # performance. The data should be prefetched and cached to device memory.
  # This will only be a problem for large datasets.
  # TODO(joetoth): Add support for prefetching and caching.
  ##############################################################################

  @property
  def dataset_keys(self):
    """Returns the keys of the datasets in the pipeline.

    Returns:
      The keys of the datasets in the pipeline.
    """
    return list(self._dataset_groups.keys())

  def get_text_encoder_n_dims(self, name: str) -> int:
    """Returns the text encoder n_dims for the given dataset name."""
    return self._modules[name].encoder.text_encoder_n_dims

  def get_numeric_n_bins(self, name: str) -> int:
    """Returns the n_bins for the given dataset name."""
    return self._modules[name].encoder.n_bins

  def get_task_info(self, name: str) -> types.SupervisedTaskInfo:
    """Returns the task info for the given dataset name."""
    return self._task_infos[name]

  def get_train_data(self, name: str) -> IndexedDataset:
    """Returns the training data for the given dataset name and step.

    Args:
      name: The name of the dataset.

    Returns:
      The training data for the given dataset name.
    """
    return self._dataset_groups[name].train

  def get_test_data(self, name: str) -> IndexedDataset:
    """Returns the test data for the given dataset name and step.

    Args:
      name: The name of the dataset.

    Returns:
      The test data for the given dataset name.
    """
    return self._dataset_groups[name].test

  def get_val_data(self, name: str) -> IndexedDataset:
    """Returns the validation data for the given dataset name and step.

    Args:
      name: The name of the dataset.

    Returns:
      The validation data for the given dataset name.
    """
    return self._dataset_groups[name].val
