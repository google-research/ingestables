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

"""Functions to encode numeric and categorical features."""

from typing import Any, Dict, Literal, Optional
import warnings

from absl import logging
from ingestables.torch.data import preprocessing
import numpy as np
from sklearn import preprocessing as sklearn_preprocessing
from sklearn import tree as sklearn_tree
import torch
import tqdm

NDArray = np.typing.NDArray

# --------------------------------------------------------------------------
# Functions to encode numeric features.
# --------------------------------------------------------------------------


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
    bins = preprocessing.Quantiles.from_sample(
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

  for column in tqdm.tqdm(
      x.T, disable=not verbose, desc="Target-aware binning:"
  ):
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
