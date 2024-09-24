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

"""Utility functions for loading and pre-processing datasets."""

import json
import os
import re
import string
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import warnings

from absl import logging
import arff
from etils import epath
from ingestables.torch import types
from ingestables.torch.data import base
import numpy as np
import pandas as pd
import scipy.io.arff as scipy_load
from sklearn import impute
from sklearn import preprocessing as sklearn_preprocessing
from sklearn import tree as sklearn_tree
import sklearn.model_selection as sklearn_model_selection
import torch
from torch.utils import data as torch_data
import tqdm

NDArray = np.typing.NDArray
ROOT_PATH = epath.Path("~/ingestables")
BASE_PATHS = {
    "carte": ROOT_PATH / "datasets/carte/preprocessed",
    "opentabs": ROOT_PATH / "datasets/opentabs",
    "ingestables": ROOT_PATH / "datasets/verticals",
}


def get_dataset_names_in_benchmark(
    benchmark_name: Literal["carte", "opentabs", "ingestables"],
) -> List[str]:
  """Get dataset names in a benchmark."""
  if benchmark_name == "carte":
    return [os.fspath(i).split("/")[-1] for i in BASE_PATHS["carte"].iterdir()]
  elif benchmark_name == "opentabs":
    return [
        os.fspath(i).split("/")[-1] for i in BASE_PATHS["opentabs"].iterdir()
    ]
  elif benchmark_name == "ingestables":
    risk_dataset_names = [
        "00_risk/00_autos",
        "00_risk/01_safe_driver",
        "00_risk/02_home_credit",
        "00_risk/03_give_me_some_credit",
        "00_risk/04_south_africa_debt",
        "00_risk/05_indonesian_telecom_delinquency",
    ]
    real_estate_dataset_names = [
        "01_real_estate/00_nyc_housing",
        "01_real_estate/01_us_real_estate",
        "01_real_estate/02_usa_housing",
        "01_real_estate/03_us_airbnb",
        "01_real_estate/04_nashville_housing",
    ]
    return risk_dataset_names + real_estate_dataset_names

  else:
    raise ValueError(f"Unknown benchmark {benchmark_name}")


# --------------------------------------------------------------------------
# Dataset Loading Functions
# --------------------------------------------------------------------------


def load_carte_dataset(
    dataset_name: str,
    do_drop_cols_with_missing_values: bool = True,
    do_drop_cols_with_one_unique_value: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
  """Load a pre-processed CARTE dataset."""
  df_path = BASE_PATHS["carte"] / dataset_name / "raw.parquet"
  config_path = BASE_PATHS["carte"] / dataset_name / "config_data.json"

  with df_path.open("rb") as f:
    data = pd.read_parquet(f)

  with config_path.open("rb") as f:
    config = json.load(f)

  # Do a quick type check of the target value
  target_key = config["target_name"]
  task = config["task"]

  target_dtype = data.dtypes.to_dict()[target_key]
  if task == "regression":
    assert target_dtype == "float"
  elif task == "classification":
    if target_dtype not in ["object", "string"]:
      warnings.warn(
          "Classification target should be either object or string, but got"
          f" {target_dtype}"
      )

    if target_dtype in ["float", "int"]:
      boolean_target = (data[target_key].nunique() == 2) and (
          set(pd.unique(data[target_key]).astype(int)) == set([0, 1])
      )
      if boolean_target:
        data[target_key] = data[target_key].replace({1.0: "true", 0.0: "false"})
        logging.info("Replacing boolean target to strings")

  # Drop columns and rows with misisng values
  if do_drop_cols_with_missing_values:
    data = drop_cols_with_missing_values(data=data)
  if do_drop_cols_with_one_unique_value:
    data = drop_cols_with_one_unique_value(data=data)

  # Preprocess categorical features
  # Prepreprocess string features
  # TODO(mononito): Think about pre-processing string and categorical features

  return data, config


# def load_ingestables_dataset(
#     dataset_name: str,
#     do_drop_cols_with_missing_values: bool = True,
#     do_drop_cols_with_one_unique_value: bool = True,
# ) -> Tuple[pd.DataFrame, Dict[str, str]]:
#   """Load a pre-processed Ingestables dataset."""
#   pass


# --------------------------------------------------------------------------
# Dataset Transformations
# Numerical features can have heterogeneous scales. These techniques are used
# to account for this heterogenity and bring the features to a common scale.
# --------------------------------------------------------------------------


def add_noise_to_numeric_features(
    data: Dict[str, pd.DataFrame],
    num_feature_keys: List[str],
    random_seed: int = 42,
    noise: float = 1e-3,
) -> Dict[str, pd.DataFrame]:
  """Add noise to numeric feature to prevent binning collapse.

  Args:
    data: Dict of dataframes comprising of train, val, test splits. Each
      dataframe is of shape [# samples, # features]
    num_feature_keys: List of numeric feature keys
    random_seed: ...
    noise: Controls the magnitude of noise added to numeric features

  Returns:
    Dataframe with noise added to numeric features
  """
  # Based on
  # https://github.com/yandex-research/rtdl-num-embeddings/blob/abf8a8b35854e4b06476bb48902096b0b58ffce2/lib/data.py#L192C13-L196C14

  train_num_arr = data["train"].loc[:, num_feature_keys].to_numpy()
  val_num_arr = data["val"].loc[:, num_feature_keys].to_numpy()
  test_num_arr = data["test"].loc[:, num_feature_keys].to_numpy()

  # Compute noise levels on training set
  stds = np.std(train_num_arr, axis=0, keepdims=True)
  noise_std = noise / np.maximum(stds, noise)

  # Add noise to train, val and test splits
  train_num_arr += noise_std * np.random.default_rng(
      random_seed
  ).standard_normal(train_num_arr.shape)
  val_num_arr += noise_std * np.random.default_rng(random_seed).standard_normal(
      val_num_arr.shape
  )
  test_num_arr += noise_std * np.random.default_rng(
      random_seed
  ).standard_normal(test_num_arr.shape)

  # Modify numeric features of existing dataframes
  data["train"].loc[:, num_feature_keys] = train_num_arr
  data["val"].loc[:, num_feature_keys] = val_num_arr
  data["test"].loc[:, num_feature_keys] = test_num_arr

  return data


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_https: bool = True,
    remove_html: bool = True,
    remove_non_alphanumeric: bool = True,
    truncate_len: Optional[int] = None,
) -> str:
  """Cleans up text from the web."""

  if lowercase:  # Lowercase text
    text = text.lower().strip()
  if remove_punctuation:  # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
  if remove_https:  # Remove https
    text = re.sub(r"https?://\S+", " ", text)
  if remove_html:  # Remove all HTML tags
    text = re.sub(r"<.*?>", " ", text)
  if remove_non_alphanumeric:  # Remove all non-alphanumeric characters
    text = re.sub(r"[^A-Za-z0-9\s]+", " ", text)
  text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines

  return text[:truncate_len]


class TorchDataWrapper(torch_data.Dataset):
  """A convenience torch dataset for embedding text data."""

  def __init__(self, data: pd.Series | list[str]):
    super().__init__()
    if isinstance(data, pd.Series):
      self.data = data.to_list()
    elif isinstance(data, list):
      self.data = data

  def __getitem__(self, index: int) -> str:
    return self.data[index]

  def __len__(self) -> int:
    return len(self.data)


def handle_text_features_with_missing_values(
    data: Dict[str, pd.DataFrame],
    feature_keys: List[str],
    nan_policy: Literal[
        "drop_rows", "most_frequent", "default_statement"
    ] = "most_frequent",
) -> Dict[str, pd.DataFrame]:
  """Handle text (categorical or string) features with missing values.

  Args:
    data: Dict of dataframes comprising of train, val, and test splits. Each
      dataframe is of shape [# samples, # features]
    feature_keys: List of feature keys
    nan_policy: How to handle features with missing values. One of "drop_rows",
      "most_frequent", and "default_statement".

  Returns:
    Dataframe without NaNs
  """
  # Filter dataframe to only contain specific features
  train_df = data["train"].loc[:, feature_keys]
  val_df = data["val"].loc[:, feature_keys]
  test_df = data["test"].loc[:, feature_keys]

  if nan_policy == "drop_rows":
    # Get rows with any NaN numeric value
    train_data_mask = ~train_df.isna().any(axis=1).to_numpy()
    val_data_mask = ~val_df.isna().any(axis=1).to_numpy()
    test_data_mask = ~test_df.isna().any(axis=1).to_numpy()

    processed_data = {
        "train": data["train"].iloc[train_data_mask, :],
        "val": data["val"].iloc[val_data_mask, :],
        "test": data["test"].iloc[test_data_mask, :],
    }

    return processed_data

  elif nan_policy == "most_frequent":
    # Simple Imputer for Object types handles None and np.NaN separately
    imputer = impute.SimpleImputer(strategy=nan_policy, missing_values=np.NaN)
    train_arr = train_df.replace({None: np.NaN}).to_numpy()
    imputer.fit(train_arr)

    train_arr = imputer.transform(train_arr)
    val_arr = imputer.transform(val_df.replace({None: np.NaN}).to_numpy())
    test_arr = imputer.transform(test_df.replace({None: np.NaN}).to_numpy())

    data["train"].loc[:, feature_keys] = train_arr
    data["val"].loc[:, feature_keys] = val_arr
    data["test"].loc[:, feature_keys] = test_arr

    return data

  elif nan_policy == "default_statement":
    data["train"].loc[:, feature_keys] = train_df.loc[:, feature_keys].fillna(
        "Missing value or description"
    )
    data["val"].loc[:, feature_keys] = val_df.loc[:, feature_keys].fillna(
        "Missing value or description"
    )
    data["test"].loc[:, feature_keys] = test_df.loc[:, feature_keys].fillna(
        "Missing value or description"
    )

    return data

  else:
    raise ValueError(f"Unknown technique {nan_policy}")


def handle_numeric_features_with_missing_values(
    data: Dict[str, pd.DataFrame],
    num_feature_keys: List[str],
    nan_policy: Literal["mean", "median", "constant", "drop_rows"] = "mean",
    fill_value: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
  """Handle rows with missing values.

  Args:
    data: Dict of dataframes comprising of train, val, and test splits. Each
      dataframe is of shape [# samples, # features]
    num_feature_keys: List of numeric feature keys
    nan_policy: How to handle numeric features with missing values. One of
      "mean", "median", "constant", and "drop_rows".
    fill_value: If nan_policy is constant, then this value is used to replace
      NaNs

  Returns:
    Dataframe without numeric NaNs
  """
  # Filter dataframe to only contain numeric features
  train_num_df = data["train"].loc[:, num_feature_keys]
  val_num_df = data["val"].loc[:, num_feature_keys]
  test_num_df = data["test"].loc[:, num_feature_keys]

  if nan_policy == "drop_rows":
    # Get rows with any NaN numeric value
    train_data_mask = ~train_num_df.isna().any(axis=1).to_numpy()
    val_data_mask = ~val_num_df.isna().any(axis=1).to_numpy()
    test_data_mask = ~test_num_df.isna().any(axis=1).to_numpy()

    processed_data = {
        "train": data["train"].iloc[train_data_mask, :],
        "val": data["val"].iloc[val_data_mask, :],
        "test": data["test"].iloc[test_data_mask, :],
    }

    return processed_data

  elif nan_policy in ["mean", "median", "constant"]:
    if nan_policy == "constant" and fill_value is None:
      raise ValueError(
          "If nan_policy is constant, then fill_value must be specified."
      )

    imputer = impute.SimpleImputer(strategy=nan_policy, fill_value=fill_value)
    imputer.fit(train_num_df.to_numpy())

    train_num_arr = imputer.transform(train_num_df.to_numpy())
    val_num_arr = imputer.transform(val_num_df.to_numpy())
    test_num_arr = imputer.transform(test_num_df.to_numpy())

    data["train"].loc[:, num_feature_keys] = train_num_arr
    data["val"].loc[:, num_feature_keys] = val_num_arr
    data["test"].loc[:, num_feature_keys] = test_num_arr

    return data

  else:
    raise ValueError(f"Unknown technique {nan_policy}")


# --------------------------------------------------------------------------
# Dataset pre-processing utils
# --------------------------------------------------------------------------
def get_missingness_ratio(data: pd.DataFrame) -> Dict[str, float]:
  """For each column in a table, get the ratio of missing values."""
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
    if np.NaN in v:
      unique_vals[k].remove(np.NaN)
  return unique_vals


def get_num_unique_values(data: pd.DataFrame) -> Dict[str, List[str]]:
  """For each column in a table, get the unique values."""
  return {col: data.loc[:, col].nunique() for col in data.columns}


def drop_cols_with_missing_values(
    data: pd.DataFrame, proportion: float = 0.5
) -> pd.DataFrame:
  """Drop columns with a high ratio of missing values."""
  missingness_ratio = get_missingness_ratio(data)
  drop_cols = [
      col for col in data.columns if missingness_ratio[col] > proportion
  ]
  return data.drop(columns=drop_cols)


def drop_cols_with_one_unique_value(data: pd.DataFrame) -> pd.DataFrame:
  """Drop columns with a single unique value."""
  num_unique_values = get_num_unique_values(data)
  drop_cols = [col for col in data.columns if num_unique_values[col] == 1]
  return data.drop(columns=drop_cols)


def infer_feature_types(
    data: pd.DataFrame, max_num_categories: int = 32
) -> Dict[str, str]:
  """Classify features into numeric, categorical, and string features."""
  feature_types_ = data.dtypes.to_dict()
  num_unique_values = get_num_unique_values(data)
  feature_types = {}
  for col, type_ in feature_types_.items():
    if feature_types_[col] == "object":
      feature_types[col] = (
          "string"
          if num_unique_values[col] > max_num_categories
          else "categorical"
      )
    elif feature_types_[col] in ["float", "int"]:
      feature_types[col] = "numeric"
    else:
      raise ValueError(f"Unknown feature {col} of type {type_}")
  return feature_types


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
  feature_types = infer_feature_types(
      data, max_num_categories=max_num_categories
  )
  unique_values = get_unique_values(data)
  for col, feat_type in feature_types.items():
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
          categories=unique_values[col],  # pytype: disable=attribute-error
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
    dataset_name: str,
    benchmark: Literal["carte", "opentabs"] = "carte",
    do_drop_cols_with_missing_values: bool = True,
    do_drop_cols_with_one_unique_value: bool = True,
    max_num_categories: int = 32,
) -> Tuple[pd.DataFrame, base.DatasetDescription]:
  """Get programmatic description of a dataset."""
  if benchmark == "carte":
    data, config = load_carte_dataset(
        dataset_name=dataset_name,
        do_drop_cols_with_missing_values=do_drop_cols_with_missing_values,
        do_drop_cols_with_one_unique_value=do_drop_cols_with_one_unique_value,
    )
  elif benchmark == "opentabs":
    # data, config = load_opentabs_dataset(dataset_name=dataset_name)
    raise NotImplementedError("Opentabs dataset is not implemented yet.")
  else:
    raise ValueError(f"Unknown benchmark {benchmark}")

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

  target_key = config["target_name"]
  if config["task"] == "classification":
    task_information = types.ClassificationTaskInfo(
        target_key=target_key,
        target_classes=feature_descriptions[target_key].categories,  # pytype: disable=attribute-error
    )
  elif config["task"] == "regression":
    task_information = types.RegressionTaskInfo(target_key=target_key)
  else:
    raise ValueError(f"Unknown task {config['task']}")

  dataset_description = base.DatasetDescription(
      dataset_name=dataset_name,
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
  return data, dataset_description


def split_dataset(
    data: pd.DataFrame,
    task_info: types.ClassificationTaskInfo | types.RegressionTaskInfo,
    train_ratio: float = 0.64,
    val_ratio: float = 0.16,
    test_ratio: float = 0.2,
    random_state: int = 13,
) -> Dict[str, pd.DataFrame]:
  """Split a dataset randomly into a train, validation, and test splits.

  Args:
    data: pd.Dataframe
    task_info: Task information
    train_ratio: float = 0.64
    val_ratio: float = 0.16
    test_ratio: float = 0.2
    random_state: int

  Returns:
    Train, validation and test data and targets.
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

  assert len(train_data) + len(val_data) + len(test_data) == len(data)
  assert np.abs((len(train_data) / len(data)) - train_ratio) < 0.1
  assert np.abs((len(val_data) / len(data)) - val_ratio) < 0.1
  assert np.abs((len(test_data) / len(data)) - test_ratio) < 0.1

  return {
      "train": train_data,
      "val": val_data,
      "test": test_data,
  }


# --------------------------------------------------------------------------
# Scaling techniques
# Numerical features can have heterogeneous scales. These techniques are used
# to account for this heterogenity and bring the features to a common scale.
# --------------------------------------------------------------------------
def scale_numeric_features(
    data: Dict[str, pd.DataFrame],
    num_feature_keys: List[str],
    scaling_method: Literal[
        "min-max", "standard", "mean", "quantile"
    ] = "quantile",
    num_quantiles: int = 48,
) -> Dict[str, pd.DataFrame]:
  """Scale numeric features.

  Args:
    data: Dict of dataframes comprising of train, val, and test splits. Each
      dataframe is of shape [# samples, # features]
    num_feature_keys: List of numeric feature keys
    scaling_method: One of "min-max", "standard", "mean".
    num_quantiles: Number of quantiles. Used in case of quantile normalization.

  Returns:
    Dataframe with out numeric NaNs
  """
  # TODO(mononito): Check issues with shallow / deep copy
  train_num_arr = data["train"].loc[:, num_feature_keys].to_numpy()
  val_num_arr = data["val"].loc[:, num_feature_keys].to_numpy()
  test_num_arr = data["test"].loc[:, num_feature_keys].to_numpy()

  ai, bi = None, None
  # NOTE: Stats are ordered as [mean, abs. mean, std, min, and max.]
  if scaling_method == "mean":
    bi = 0
    ai = np.nanmean(np.abs(train_num_arr), axis=0, keepdims=True)
  elif scaling_method == "min-max":
    bi = np.nanmin(np.abs(train_num_arr), axis=0, keepdims=True)
    ai = np.nanmax(np.abs(train_num_arr), axis=0, keepdims=True) - bi
  elif scaling_method == "standard":
    bi = np.nanmean(train_num_arr, axis=0, keepdims=True)
    ai = np.nanstd(train_num_arr, axis=0, keepdims=True)
  elif scaling_method == "quantile":
    # Compute quantiles based on the train split
    quantiles = Quantiles.from_sample(
        num_quantiles=num_quantiles,
        numeric_feature_values_raw=train_num_arr,
        numeric_feature_keys=num_feature_keys,
    ).as_ndarray()

    max_quantiles_plus_one = quantiles.shape[1]
    num_quantiles_plus_one = max_quantiles_plus_one - np.isnan(quantiles).sum(
        axis=-1
    )
    # NOTE: The number of quantiles is not fixed. Different features can have
    # different numbers of quantiles.
    num_numeric_feats = len(num_feature_keys)

    # Quantile normalize train, val and test arrays
    train_num_arr = np.stack(
        [
            np.interp(
                train_num_arr[:, i],
                xp=quantiles[i, : num_quantiles_plus_one[i]],
                fp=np.linspace(-1, 1, num=num_quantiles_plus_one[i]),
            )
            for i in range(num_numeric_feats)
        ],
        axis=1,
    )
    val_num_arr = np.stack(
        [
            np.interp(
                val_num_arr[:, i],
                xp=quantiles[i, : num_quantiles_plus_one[i]],
                fp=np.linspace(-1, 1, num=num_quantiles_plus_one[i]),
            )
            for i in range(num_numeric_feats)
        ],
        axis=1,
    )
    test_num_arr = np.stack(
        [
            np.interp(
                test_num_arr[:, i],
                xp=quantiles[i, : num_quantiles_plus_one[i]],
                fp=np.linspace(-1, 1, num=num_quantiles_plus_one[i]),
            )
            for i in range(num_numeric_feats)
        ],
        axis=1,
    )

  else:
    raise ValueError(f"Unsupported scaling_method: {scaling_method=}")

  if scaling_method != "quantile":
    train_num_arr = (train_num_arr - bi) / ai
    val_num_arr = (val_num_arr - bi) / ai
    test_num_arr = (test_num_arr - bi) / ai

  data["train"].loc[:, num_feature_keys] = train_num_arr
  data["val"].loc[:, num_feature_keys] = val_num_arr
  data["test"].loc[:, num_feature_keys] = test_num_arr

  return data


class Quantiles:
  """Convenience class for dealing with quantiles."""

  def __init__(
      self,
      num_quantiles: int,
      quantiles_dict: Dict[str, np.ndarray],
      numeric_feature_keys: List[str],
  ):
    self._num_quantiles = num_quantiles
    self._numeric_feature_keys = numeric_feature_keys
    self._quantiles_dict = quantiles_dict

  def __repr__(self):
    return (
        f"Quantiles(num_quantiles={self._num_quantiles},"
        f" numeric_feature_keys={self._numeric_feature_keys},"
        f" quantiles_dict={self._quantiles_dict})"
    )

  @classmethod
  def from_sample(
      cls,
      num_quantiles: int,
      numeric_feature_values_raw: np.ndarray,
      numeric_feature_keys: Optional[List[str]] = None,
  ) -> "Quantiles":
    """Create a Quantiles object from a sample batch.

    Args:
      num_quantiles: Number of quantile buckets. Note that the number of
        quantile boundaries is num_quantiles + 1.
      numeric_feature_values_raw: np.ndarray of numeric features, of shape
        [batch_size, num_numeric_features]. Note that the numeric features along
        axis 1 must be correspond to the order in numeric_feature_keys.
      numeric_feature_keys: List of numeric feature keys.

    Returns:
      The Quantiles object.
    """
    if numeric_feature_values_raw.ndim == 1:
      numeric_feature_values_raw = numeric_feature_values_raw[None, ...]

    if numeric_feature_keys is None:
      numeric_feature_keys = [
          f"num_feat_{i}" for i in range(numeric_feature_values_raw.shape[1])
      ]

    quantiles_arr = np.quantile(
        numeric_feature_values_raw,
        q=np.linspace(
            start=0.0, stop=1.0, num=num_quantiles + 1, endpoint=True
        ),
        axis=1,
    ).astype(np.float32)

    quantiles_arr = np.transpose(quantiles_arr)

    for i in range(quantiles_arr.shape[0]):
      unique_quantiles = np.unique(quantiles_arr[i, :])
      padding = 1 + num_quantiles - len(unique_quantiles)
      if padding > 0:
        quantiles_arr[i, :] = np.pad(
            unique_quantiles,
            (0, padding),
            mode="constant",
            constant_values=np.nan,
        )
        # Right pad quantile arrays to be of shape 1 + num_quantiles

    quantiles_dict = {
        num_feat_key: num_feat_quantiles
        for num_feat_key, num_feat_quantiles in zip(
            numeric_feature_keys, list(quantiles_arr)
        )
    }

    return Quantiles(
        num_quantiles=num_quantiles,
        numeric_feature_keys=numeric_feature_keys,
        quantiles_dict=quantiles_dict,
    )

  @property
  def num_quantiles(self) -> int:
    return self._num_quantiles

  @property
  def numeric_feature_keys(self) -> List[str]:
    return self._numeric_feature_keys

  def as_ndarray(
      self,
      numeric_feature_keys: Optional[List[str]] = None,
  ) -> np.ndarray:
    """Returns ndarray of shape [num_numeric_features, num_quantiles]."""
    numeric_feature_keys = numeric_feature_keys or self._numeric_feature_keys
    quantiles_list = [
        self._quantiles_dict[numeric_feature_key]
        for numeric_feature_key in numeric_feature_keys
    ]
    quantiles_arr = np.stack(quantiles_list, axis=0)

    assert quantiles_arr.shape == (
        len(numeric_feature_keys),
        self._num_quantiles + 1,
    )
    return quantiles_arr


# -------------------------------------------------------   -------------------
# Encoding techniques
# Given a scalar numeric feature, an numeric encoder converts it to a vector of
# length n_bins (number of bins). This encoding may have NaNs if the computed
# bins are fewer than n_bins.
# --------------------------------------------------------------------------
def soft_one_hot_encoding(
    x: torch.Tensor,
    edges: torch.Tensor,
):
  """Performs soft one-hot encoding of the input features.

  Args:
      x: A tf.Tensor of shape [n_observations, n_features].
      edges: A tf.Tensor of shape [n_features, n_bins + 1] containing the bin
        edges.

  Returns:
      A tf.Tensor of shape [n_observations, n_features, n_bins] containing the
      soft one-hot encoded features.
  """

  bin_centers = (edges[:, :-1] + edges[:, 1:]) / 2  # Calculate bin centers
  std = 1 / torch.tensor(bin_centers.shape[0]).float()  # Standard deviation

  # Calculate z-score (normalized distance from bin centers)
  z_score = (x.unsqueeze(-1) - bin_centers.unsqueeze(0)) / std

  # Replace NaNs with zeros
  z_score = torch.where(
      torch.isnan(z_score), torch.zeros_like(z_score), z_score
  )

  # Apply softmax with squared negative z-score for soft assignment
  return torch.nn.functional.softmax(-torch.square(z_score), dim=-1)


def piecewise_linear_encoding(
    x: torch.Tensor,
    edges: torch.Tensor,
):
  """Performs piecewise linear encoding on numeric features based on bin edges.

  Args:
    x: A tf.Tensor of shape [n_observations, n_features].
    edges: A tf.Tensor of shape [n_features, n_bins + 1] containing the bin
      edges.

  Returns:
    A tf.Tensor of shape [n_observations, n_features, n_bins] containing the
    piecewise linear encoding.
  """

  left_edges = edges[:, :-1]
  width = edges[:, 1:] - edges[:, :-1]

  bin_counts = torch.sum(
      torch.where(
          torch.isnan(edges),
          torch.zeros_like(edges).int(),
          torch.ones_like(edges).int(),
      ),
      axis=1,
  ).numpy()

  # x: [n_observations, n_features]
  x = (x.unsqueeze(-1) - left_edges.unsqueeze(0)) / width.unsqueeze(0)
  # x: [n_observations, n_features, n_bins]

  n_bins = x.shape[-1]
  # Piecewise linear encoding with clipping for boundaries
  ple = []
  for i, count in enumerate(bin_counts):
    if count == 1:
      ple.append(x[..., i, :])
    else:
      clipped = torch.cat(
          [
              x[..., i, :1].clamp_max(1.0),
              *(
                  []
                  if n_bins == 2
                  else [x[..., i, 1 : count - 1].clamp(0.0, 1.0)]
              ),
              x[..., i, count - 1 : count].clamp_min(0.0),
              x[..., i, count:],
          ],
          dim=-1,
      )
      ple.append(clipped)
  encoding = torch.stack(ple, dim=-2)
  return torch.where(
      torch.isnan(encoding), torch.zeros_like(encoding), encoding
  )


# --------------------------------------------------------------------------
# Binning functions.
# --------------------------------------------------------------------------
def compute_bins(
    numeric_feature_values_raw: np.ndarray,
    n_bins: int = 48,
    binning_method: Literal["target-aware", "quantile", "uniform"] = "quantile",
    tree_kwargs: Optional[Dict[str, Any]] = None,
    target_values: Optional[np.ndarray] = None,
    task: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
  """Compute bin edges for `PiecewiseLinearEmbeddings`.

  Args:
    numeric_feature_values_raw: Array of un-normalized numeric features, of
      shape [batch_size, num_numeric_features]. Note that the numeric features
      along axis 1 must be correspond to the order in numeric_feature_keys.
    n_bins: the number of bins.
    binning_method: How to compute the bin edges. One of "target-aware",
      "quantile", or "uniform".
    tree_kwargs: keyword arguments for `sklearn.tree.DecisionTreeRegressor` (if
      ``task`` is `regression``), or `sklearn.tree.DecisionTreeClassifier` (if
      ``task`` is `classification`).
    target_values: the training labels (must be provided if ``tree`` is not
      None).
    task: Whether a regression or a classification task.
    verbose: controls verbosity.

  Returns:
    A list of bin edges for all features. For one feature:

    - the maximum possible number of bin edges is ``n_bins + 1``.
    - the minimum possible number of bin edges is ``1``.
  """
  if np.ndim(target_values) != 1:
    raise ValueError(
        "target_values must have exactly one dimension, however:"
        + f" {np.ndim(target_values)=}"
    )
  if len(target_values) != len(numeric_feature_values_raw):
    raise ValueError(
        "len(target_values) must be equal to len(X), however:"
        + f" {len(target_values)=}, {len(numeric_feature_values_raw)=}"
    )
  if target_values is None or task is None:
    raise ValueError(
        "If tree_kwargs is not None, then target_values and task must not be"
        + " None"
    )

  if binning_method == "quantile":
    bins = Quantiles.from_sample(
        num_quantiles=n_bins,
        numeric_feature_values_raw=numeric_feature_values_raw,
        numeric_feature_keys=None,
    ).as_ndarray()
  elif binning_method == "uniform":
    bins = compute_uniform_bins(numeric_feature_values_raw, n_bins)
  elif binning_method == "target-aware":
    bins = compute_target_aware_bins(
        numeric_feature_values_raw,
        n_bins,
        tree_kwargs,
        target_values,
        task,
        verbose,
    )
  else:
    raise ValueError(
        f"Unsupported binning_method: {binning_method=}. "
        "Supported values are: 'quantile', 'uniform', 'target-aware'"
    )
  validate_bins(bins, suppress_warnings=True)

  return bins


def compute_target_aware_bins(
    x: NDArray,
    n_bins: int = 48,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    target_values: Optional[NDArray] = None,
    task: Optional[Literal["classification", "regression"]] = None,
    verbose: bool = False,
) -> NDArray:
  """Compute target-aware bin edges.

  Args:
    x: training features of shape [num_observations, num_numeric_features].
    n_bins: the number of bins.
    tree_kwargs: keyword arguments for `sklearn.tree.DecisionTreeRegressor` (if
      ``task`` is `regression``), or `sklearn.tree.DecisionTreeClassifier` (if
      ``task`` is `classification`).
    target_values: the training labels (must be provided if ``tree`` is not
      None).
    task: Whether a regression or a classification task.
    verbose: controls verbosity.

  Returns:
    An array of bin edges for all features of shape [num_numeric_features,
    n_bins + 1].
  """
  le = sklearn_preprocessing.LabelEncoder()
  target_values = le.fit_transform(target_values)

  if tree_kwargs is None:
    tree_kwargs = {}
  bins = []

  for column in tqdm.tqdm(x.T, disable=not verbose):
    feature_bin_edges = [float(column.min()), float(column.max())]
    tree = (
        (
            sklearn_tree.DecisionTreeRegressor
            if task == "regression"
            else sklearn_tree.DecisionTreeClassifier
        )(max_leaf_nodes=n_bins, **tree_kwargs)
        .fit(column.reshape(-1, 1), target_values)
        .tree_
    )
    for node_id in range(tree.node_count):
      # The following condition is True only for split nodes. Source:
      # https://scikit-learn.org/1.0/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
      if tree.children_left[node_id] != tree.children_right[node_id]:
        feature_bin_edges.append(float(tree.threshold[node_id]))

    bins_ = np.sort(np.unique(feature_bin_edges))
    if len(bins_) < n_bins + 1:
      bins_ = np.pad(
          bins_,
          (0, 1 + n_bins - len(bins_)),
          mode="constant",
          constant_values=np.nan,
      )
    bins.append(bins_)

  return np.array(bins)


def compute_uniform_bins(x: NDArray, n_bins: int = 48) -> NDArray:
  """Compute uniform bin edges.

  Args:
    x: training features of shape (num_observations, num_numeric_features)
    n_bins: the number of bins.

  Returns:
    An array of bin edges for all features of shape [num_numeric_features,
    n_bins + 1].
  """

  n_features = x.shape[1]
  mins = x.min(axis=0)
  maxs = x.max(axis=0)
  return np.stack(
      [np.linspace(mins[i], maxs[i], n_bins + 1) for i in range(n_features)]
  )


def validate_bins(bins: NDArray, suppress_warnings: bool = False) -> None:
  """Function to if bins are valid."""
  if suppress_warnings:
    logging.info("Some warnings are suppressed.")
  if len(bins) == 0:  # pylint: disable=g-explicit-length-test
    raise ValueError("The list of bins must not be empty")
  for i, feature_bins in enumerate(bins):
    if feature_bins.ndim != 1:
      raise ValueError(
          "Each item of the bin list must have exactly one dimension."
          f" However, for {i=}: {bins[i].ndim=}"
      )
    if len(feature_bins) < 2:
      raise ValueError(
          "All features must have at least two bin edges."
          f" However, for {i=}: {len(bins[i])=}"
      )
    if not np.isfinite(feature_bins).all() and not suppress_warnings:
      warnings.warn(
          "Bin edges must not contain nan/inf/-inf."
          f" However, this is not true for the {i}-th feature."
          " This may be because of computed bins < n_bins"
      )
    if (feature_bins[:-1] >= feature_bins[1:]).any():
      raise ValueError(
          f"Bin edges must be sorted. However, the for the {i}-th feature, the"
          + f" bin edges {feature_bins} and  are not sorted"
      )
    if len(feature_bins) == 2:
      warnings.warn(
          f"The {i}-th feature has just two bin edges, which means only one"
          " bin. Strictly speaking, using a single bin for the"
          " piecewise-linear encoding should not break anything, but it is the"
          " same as using sklearn.preprocessing.MinMaxScaler"
      )


def read_arff(
    vertical_name: str = "00_risk",
    dataset_name: str = "04_south_africa_debt",
) -> tuple[np.ndarray, scipy_load.MetaData]:
  """Function to read arff dataset."""

  path = (
      epath.Path(BASE_PATHS["ingestables"])
      / vertical_name
      / dataset_name
      / "raw_dataset.arff"
  )

  try:
    with path.open(mode="r") as f:
      data, meta = scipy_load.loadarff(f)
  except NotImplementedError as exp:
    with path.open(mode="r") as f:
      print(f"Failed to load: {exp}. Trying to load with arff.load(...)")

      del f
      with path.open(mode="r") as f:
        data_and_metadata = arff.load(f)

      attr = []
      # TODO(mononito): Import from scipy_load.arff.to_attribute does not work
      # but arffread is going to be deprecated.
      for attr_name, attr_type in data_and_metadata["attributes"]:
        attr.append(scipy_load.arffread.to_attribute(attr_name, attr_type))

      meta = scipy_load.MetaData(rel=data_and_metadata["relation"], attr=attr)

      data = []
      for i in range(len(data_and_metadata["data"])):
        data.append(
            tuple([data_and_metadata["data"][i][j] for j in range(len(attr))])
        )

      data = np.array(data, dtype=([(a.name, a.dtype) for a in attr]))

  return data, meta
