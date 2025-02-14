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

"""Pre-processing functions for data."""

import re
import string
from typing import Any, Dict, List, Literal, Optional
import warnings

from absl import logging
from ingestables.torch.data import utils
import numpy as np
import pandas as pd
from sklearn import impute
import sklearn.preprocessing as sklearn_preprocessing
import sklearn.tree as sklearn_tree
import torch
import tqdm


NumericNanPolicy = Literal["mean", "median", "constant", "drop_rows"]
NumericScalingMethod = Literal[
    "min-max", "standard", "mean", "quantile", "power"
]
StringNanPolicy = Literal["drop_rows", "default_statement"]
CategoricalNanPolicy = Literal[
    "drop_rows", "most_frequent", "default_statement"
]


# --------------------------------------------------------------------------
# Dataset Transformations
# Numerical features can have heterogeneous scales. These techniques are used
# to account for this heterogenity and bring the features to a common scale.
# --------------------------------------------------------------------------


def drop_cols_with_missing_values(
    data: Dict[str, pd.DataFrame], proportion: float = 0.5
) -> Dict[str, pd.DataFrame]:
  """Drop columns with a high ratio of missing values."""

  def _drop_cols(data: pd.DataFrame) -> List[str]:
    missingness_ratio = utils.get_missingness_ratio(data)
    return [col for col in data.columns if missingness_ratio[col] > proportion]

  # if isinstance(data, pd.DataFrame):
  #   return data.drop(columns=_drop_cols(data))

  # TODO(mononito): "train", "val", "test" are hardcoded.
  drop_cols_train = set(_drop_cols(data["train"]))
  drop_cols_val = set(_drop_cols(data["val"]))
  drop_cols_test = set(_drop_cols(data["test"]))
  drop_cols = drop_cols_train.intersection(drop_cols_val, drop_cols_test)
  return {k: v.drop(columns=drop_cols) for k, v in data.items()}


def drop_cols_with_one_unique_value(
    data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
  """Drop columns with a single unique value."""

  def _drop_cols(data: pd.DataFrame) -> List[str]:
    num_unique_values = utils.get_num_unique_values(data)
    return [col for col in data.columns if num_unique_values[col] == 1]

  # if isinstance(data, pd.DataFrame):
  #   return data.drop(columns=_drop_cols(data))

  drop_cols_train = set(_drop_cols(data["train"]))
  drop_cols_val = set(_drop_cols(data["val"]))
  drop_cols_test = set(_drop_cols(data["test"]))
  drop_cols = drop_cols_train.intersection(drop_cols_val, drop_cols_test)
  return {k: v.drop(columns=drop_cols) for k, v in data.items()}


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

  train_num_arr = (
      data["train"].loc[:, num_feature_keys].to_numpy().astype(np.float32)
  )
  val_num_arr = (
      data["val"].loc[:, num_feature_keys].to_numpy().astype(np.float32)
  )
  test_num_arr = (
      data["test"].loc[:, num_feature_keys].to_numpy().astype(np.float32)
  )

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


def handle_text_features_with_missing_values(
    data: Dict[str, pd.DataFrame],
    feature_keys: List[str],
    nan_policy: CategoricalNanPolicy | StringNanPolicy = "most_frequent",
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
    imputer = impute.SimpleImputer(strategy=nan_policy, missing_values=np.nan)
    train_arr = train_df.replace({None: np.nan}).to_numpy()
    val_arr = val_df.replace({None: np.nan}).to_numpy()
    test_arr = test_df.replace({None: np.nan}).to_numpy()

    imputer.fit(np.concatenate([train_arr, val_arr], axis=0))

    train_arr = imputer.transform(train_arr)
    val_arr = imputer.transform(val_arr)
    test_arr = imputer.fit_transform(test_arr)

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
    nan_policy: NumericNanPolicy = "mean",
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
# Scaling techniques
# Numerical features can have heterogeneous scales. These techniques are used
# to account for this heterogenity and bring the features to a common scale.
# --------------------------------------------------------------------------
def scale_numeric_features(
    data: Dict[str, pd.DataFrame],
    num_feature_keys: List[str],
    scaling_method: NumericScalingMethod = "quantile",
    num_quantiles: int = 48,
    quantile_transformer_method: Literal["sklearn", "custom"] = "custom",
    quantile_transform_subsample: int = 10_000,
    quantile_transform_output_distribution: str = "normal",
) -> Dict[str, pd.DataFrame]:
  """Scale numeric features.

  Args:
    data: Dict of dataframes comprising of train, val, and test splits. Each
      dataframe is of shape [# samples, # features]
    num_feature_keys: List of numeric feature keys
    scaling_method: One of "min-max", "standard", "mean".
    num_quantiles: Number of quantiles. Used in case of quantile normalization.
    quantile_transformer_method: Method to use for quantile transformation. One
      of "sklearn" or "custom". "sklearn" uses Scikit-Learn's
      QuantileTransformer while "custom" uses a custom implementation.
    quantile_transform_subsample: Subsample size for quantile transformation,
      used by Scikit-Learn's QuantileTransformer.
    quantile_transform_output_distribution: Output distribution for quantile,
      used by Scikit-Learn's QuantileTransformer.

  Returns:
    Dataframe with scaled numeric features
  """
  # TODO(mononito): Check issues with shallow / deep copy
  train_num_arr = data["train"].loc[:, num_feature_keys].to_numpy()
  val_num_arr = data["val"].loc[:, num_feature_keys].to_numpy()
  test_num_arr = data["test"].loc[:, num_feature_keys].to_numpy()

  ai, bi = None, None
  if scaling_method == "mean":
    bi = 0
    ai = np.nanmean(np.abs(train_num_arr), axis=0, keepdims=True)
  elif scaling_method == "min-max":
    bi = np.nanmin(np.abs(train_num_arr), axis=0, keepdims=True)
    ai = np.nanmax(np.abs(train_num_arr), axis=0, keepdims=True) - bi
  elif scaling_method == "standard":
    bi = np.nanmean(train_num_arr, axis=0, keepdims=True)
    ai = np.nanstd(train_num_arr, axis=0, keepdims=True)
  elif scaling_method == "power":
    power_transformer = sklearn_preprocessing.PowerTransformer(
        method="yeo-johnson", standardize=True, copy=True
    )
    train_num_arr = power_transformer.fit_transform(train_num_arr)
    val_num_arr = power_transformer.transform(val_num_arr)
    test_num_arr = power_transformer.transform(test_num_arr)
  elif (
      scaling_method == "quantile" and quantile_transformer_method == "sklearn"
  ):
    quantile_transformer = sklearn_preprocessing.QuantileTransformer(
        n_quantiles=num_quantiles,
        output_distribution=quantile_transform_output_distribution,
        subsample=quantile_transform_subsample,
        copy=True,
    )
    train_num_arr = quantile_transformer.fit_transform(train_num_arr)
    val_num_arr = quantile_transformer.transform(val_num_arr)
    test_num_arr = quantile_transformer.transform(test_num_arr)
  elif scaling_method == "quantile" and quantile_transformer_method == "custom":
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

  if scaling_method in ["mean", "min-max", "standard"]:
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
    x: np.ndarray,
    n_bins: int = 48,
    tree_kwargs: Optional[Dict[str, Any]] = None,
    target_values: Optional[np.ndarray] = None,
    task: Optional[Literal["classification", "regression"]] = None,
    verbose: bool = False,
) -> np.ndarray:
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


def compute_uniform_bins(x: np.ndarray, n_bins: int = 48) -> np.ndarray:
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


def validate_bins(
    bins: np.ndarray | torch.Tensor, suppress_warnings: bool = False
) -> None:
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
