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

"""Utility functions for data processing and understanding."""

from typing import Dict, List, Tuple, Union
from ingestables.torch import types
from ingestables.torch.data import base
import numpy as np
import pandas as pd
from torch.utils import data as torch_data


class TorchDataWrapper(torch_data.Dataset):
  """A convenience torch dataset for embedding text data."""

  def __init__(self, data: pd.Series | list[str] | np.ndarray):
    super().__init__()
    if isinstance(data, pd.Series):
      self.data = data.to_list()
    elif isinstance(data, list):
      self.data = data
    elif isinstance(data, np.ndarray):
      if data.ndim != 1:
        raise ValueError("Data must be 1-dimensional")
      self.data = data.tolist()

  def __getitem__(self, index: int) -> str:
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)


def get_missingness_ratio(data: pd.DataFrame) -> Dict[str, float]:
  """For each column in a table, get the ratio of missing values.

  Args:
    data: Dataframe

  Returns:
    A dictionary mapping column names to the ratio of missing values in that
    column.
  """
  return data.isnull().astype(int).mean(axis=0).to_dict()


def get_unique_values(data: pd.DataFrame) -> Dict[str, List[str]]:
  """For each column in a table, get the unique values."""
  unique_vals = {
      col: pd.unique(data.loc[:, col]).tolist() for col in data.columns
  }
  # Remove NaN and None as unique values
  for k, v in unique_vals.items():
    if None in v:
      unique_vals[k].remove(None)
    if np.nan in v:
      unique_vals[k].remove(np.nan)
  return unique_vals


def get_num_unique_values(data: pd.DataFrame) -> Dict[str, List[str]]:
  """For each column in a table, get the unique values."""
  return {col: data.loc[:, col].nunique() for col in data.columns}


def infer_feature_types(
    data: pd.DataFrame, max_num_categories: int = 32
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
  """Classify features into numeric, categorical, and string features.

  Args:
    data: Dataframe
    max_num_categories: Maximum number of categories for categorical features

  Returns:
    A tuple containing two dictionaries:
      - feature_to_type: A dictionary mapping feature names to their inferred
        types ("numeric", "categorical", or "string").
      - feature_type_dict: A dictionary mapping feature types to a list of
        feature names of that type.
  """
  feature_types_ = data.dtypes.to_dict()
  num_unique_values = get_num_unique_values(data)

  feature_to_type = {}
  feature_type_dict = {"categorical": [], "numeric": [], "string": []}
  for col, type_ in feature_types_.items():
    if feature_types_[col] == "object":
      if num_unique_values[col] <= max_num_categories:
        feature_to_type[col] = "categorical"
        feature_type_dict["categorical"].append(col)
      else:
        feature_to_type[col] = "string"
        feature_type_dict["string"].append(col)
    elif pd.api.types.is_any_real_numeric_dtype(data[col]):
      feature_to_type[col] = "numeric"
      feature_type_dict["numeric"].append(col)
    else:
      raise ValueError(f"Unknown feature {col} of type {type_}")

  return feature_to_type, feature_type_dict


def get_examples(
    data: Union[pd.DataFrame, pd.Series],
    num_examples: int = 5,
    random_state: int = 13,
) -> List[str]:
  """Get examples from a dataframe.

  This function is used to give examples of rows in a table or columns.

  Args:
    data: A feature or a table
    num_examples: Number of examples
    random_state: integer to control randomness

  Returns:
    A list of examples.
  """
  examples = data.sample(frac=1, random_state=random_state).iloc[:num_examples]
  return [row for row in examples]


def get_feature_descriptions(
    data: pd.DataFrame,
    num_examples: int = 5,
    random_state: int = 13,
    max_num_categories: int = 32,
) -> Dict[str, base.FeatureDescription]:
  """Get feature descriptions from a table.

  Args:
    data: Dataframe
    num_examples: Number of examples for the summary
    random_state: Random state
    max_num_categories: Maximum number of categorical values

  Returns:
    List of feature descriptions
  """
  feature_descriptions = {}
  feature_to_type, _ = infer_feature_types(
      data, max_num_categories=max_num_categories
  )
  unique_values = get_unique_values(data)
  for col, feat_type in feature_to_type.items():
    if feat_type == "string":
      string_lengths = [len(i) for i in data[col] if i is not None]
      feature_descriptions[col] = base.StringFeatureDescription(
          feature_name=col,
          max_length=max(string_lengths),
          min_length=min(string_lengths),
          example_strings=get_examples(data[col], num_examples, random_state),
      )
    elif feat_type == "categorical":
      feature_descriptions[col] = base.CategoricalFeatureDescription(
          feature_name=col,
          num_categories=len(unique_values[col]),
          categories=unique_values[col],
      )
    elif feat_type == "numeric":
      stats = data[col].describe().to_dict()
      feature_descriptions[col] = base.NumericFeatureDescription(
          feature_name=col,
          max=stats["max"],
          min=stats["min"],
          mean=stats["mean"],
          std=stats["std"],
          median=stats["50%"],
      )
    else:
      raise ValueError(f"Unknown feature type {type}")
  return feature_descriptions


def get_dataset_and_description(
    data: pd.DataFrame,
    task_information: types.TaskInfo,
    max_num_categories: int = 32,
) -> base.DatasetDescription:
  """Get programmatic description of a dataset.

  Args:
    data: Dataframe
    task_information: Task information.
    max_num_categories: Maximum number of values. Textual features with more
      than this number of values are considered strings.

  Returns:
    Dataset description
  """

  num_rows, num_features = data.shape
  feature_descriptions = get_feature_descriptions(data, max_num_categories)

  feat_keys_dict = {"numeric": [], "categorical": [], "string": []}
  for feature_name, desc in feature_descriptions.items():
    feat_keys_dict[desc.feature_type].append(feature_name)

  # Sort feat_keys_dict
  for k, v in feat_keys_dict.items():
    feat_keys_dict[k] = sorted(v)

  num_string_features = len(feat_keys_dict["string"])
  num_categorical_features = len(feat_keys_dict["categorical"])
  num_numeric_features = len(feat_keys_dict["numeric"])

  dataset_description = base.DatasetDescription(
      dataset_name=task_information.dataset_name,
      dataset_description=None,
      num_rows=num_rows,
      num_features=num_features,
      num_string_features=num_string_features,
      num_categorical_features=num_categorical_features,
      num_numeric_features=num_numeric_features,
      task_information=task_information,
      feature_descriptions=feature_descriptions,
      feature_keys_dict=feat_keys_dict,
  )
  return dataset_description
