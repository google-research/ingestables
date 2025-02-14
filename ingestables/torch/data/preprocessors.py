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

"""Dataset Pre-processing functions."""

import functools
from typing import Dict, Literal, Optional
import warnings
from ingestables.torch.data import preprocessing
from ingestables.torch.data import utils
import pandas as pd


class Preprocessor:
  """This pre-processing function pre-processes numerical, categorical, and string features."""

  def __init__(
      self,
      drop_cols_with_missing_values: bool = True,
      missingness_proportion: float = 0.5,
      drop_cols_with_one_unique_value: bool = True,
      max_num_categories: int = 8,
      numeric_nan_policy: preprocessing.NumericNanPolicy = "mean",
      fill_value: float = 0,
      noise: float = 1e-3,
      numeric_scaling_method: preprocessing.NumericScalingMethod = "quantile",
      num_quantiles: int = 48,
      quantile_transformer_method: Literal["sklearn", "custom"] = "sklearn",
      quantile_transform_subsample: int = 10_000,
      quantile_transform_output_distribution: str = "normal",
      string_nan_policy: preprocessing.StringNanPolicy = "default_statement",
      categorical_nan_policy: preprocessing.CategoricalNanPolicy = "most_frequent",
      lowercase: bool = True,
      remove_punctuation: bool = True,
      remove_https: bool = True,
      remove_html: bool = True,
      remove_non_alphanumeric: bool = True,
      truncate_len: int = 100,
  ):
    """Initialize the preprocessor.

    Args:
        drop_cols_with_missing_values: Whether to drop columns with missing
          values.
        missingness_proportion: The proportion of missing values to drop.
        drop_cols_with_one_unique_value: Whether to drop columns with only one
          unique value.
        max_num_categories: The maximum number of categories to use for
          categorical features.
        numeric_nan_policy: The policy to use for handling missing values in
          numeric features.
        fill_value: The value to fill missing values in numeric features.
        noise: The amount of noise to add to numeric features.
        numeric_scaling_method: The method to use for scaling numeric features.
        num_quantiles: The number of quantiles to use for quantile scaling.
        quantile_transformer_method: The method to use for quantile
          transformation.
        quantile_transform_subsample: The subsample size to use for quantile
          transformation.
        quantile_transform_output_distribution: The output distribution to use
          for quantile transformation.
        string_nan_policy: The policy to use for handling missing values in
          string features.
        categorical_nan_policy: The policy to use for handling missing values in
          categorical features.
        lowercase: Whether to lowercase textual features.
        remove_punctuation: Whether to remove punctuation from textual features.
        remove_https: Whether to remove https from textual features.
        remove_html: Whether to remove html from textual features.
        remove_non_alphanumeric: Whether to remove non-alphanumeric characters
          from textual features.
        truncate_len: The maximum length of textual features.
    """
    self.drop_cols_with_missing_values = drop_cols_with_missing_values
    self.missingness_proportion = missingness_proportion
    self.drop_cols_with_one_unique_value = drop_cols_with_one_unique_value
    self.max_num_categories = max_num_categories
    self.numeric_nan_policy = numeric_nan_policy
    self.fill_value = fill_value
    self.noise = noise
    self.numeric_scaling_method = numeric_scaling_method
    self.num_quantiles = num_quantiles
    self.quantile_transformer_method = quantile_transformer_method
    self.quantile_transform_subsample = quantile_transform_subsample
    self.quantile_transform_output_distribution = (
        quantile_transform_output_distribution
    )
    self.string_nan_policy = string_nan_policy
    self.categorical_nan_policy = categorical_nan_policy
    self.lowercase = lowercase
    self.remove_punctuation = remove_punctuation
    self.remove_https = remove_https
    self.remove_html = remove_html
    self.remove_non_alphanumeric = remove_non_alphanumeric
    self.truncate_len = truncate_len

    if (
        self.numeric_scaling_method == "quantile"
        and self.quantile_transformer_method == "custom"
    ):
      warnings.warn(
          "Quantile scaling by setting quantile_transformer_method to custom"
          + " will be deprecated.",
          DeprecationWarning,
      )

  def __call__(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Pre-processes a dictionary of dataframes.

    Args:
        data (dict): A dictionary containing the train, val, and test
          dataframes.

    Returns:
        dict: A dictionary containing the pre-processed train, val, and test
          dataframes.
    """

    if self.drop_cols_with_missing_values:
      data = preprocessing.drop_cols_with_missing_values(
          data, self.missingness_proportion
      )
    if self.drop_cols_with_one_unique_value:
      data = preprocessing.drop_cols_with_one_unique_value(data)

    # Infer feature types
    _, feature_type_dict = utils.infer_feature_types(
        data["train"], self.max_num_categories
    )
    self.categorical_features = feature_type_dict["categorical"]
    self.string_features = feature_type_dict["string"]
    self.numeric_features = feature_type_dict["numeric"]

    if len(self.numeric_features) > 0:  # pylint: disable=g-explicit-length-test
      data = self._preprocess_numeric_features(data)
    if len(self.string_features + self.categorical_features) > 0:  # pylint: disable=g-explicit-length-test
      data = self._preprocess_textual_features(data)

    return data

  def __repr__(self):
    attribute_str = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
    return f"Preprocessor({attribute_str})"

  def _preprocess_numeric_features(
      self,
      data: Dict[str, pd.DataFrame],
      numeric_features: Optional[list[str]] = None,
  ) -> Dict[str, pd.DataFrame]:
    """Pre-process numeric features."""

    if numeric_features is None:
      numeric_features = self.numeric_features
    if len(numeric_features) == 0:  # pylint: disable=g-explicit-length-test
      return data

    # 1. Handle missing values
    data = preprocessing.handle_numeric_features_with_missing_values(
        data=data,
        num_feature_keys=numeric_features,
        nan_policy=self.numeric_nan_policy,
        fill_value=self.fill_value,
    )

    # 2. Add noise to numeric features to prevent binning collapse
    data = preprocessing.add_noise_to_numeric_features(
        data=data,
        num_feature_keys=numeric_features,
        noise=self.noise,
    )

    # 3. Scale numeric features
    data = preprocessing.scale_numeric_features(
        data=data,
        num_feature_keys=numeric_features,
        scaling_method=self.numeric_scaling_method,
        num_quantiles=self.num_quantiles,
        quantile_transformer_method=self.quantile_transformer_method,
        quantile_transform_subsample=self.quantile_transform_subsample,
        quantile_transform_output_distribution=self.quantile_transform_output_distribution,
    )
    return data

  def _preprocess_textual_features(
      self,
      data: Dict[str, pd.DataFrame],
      string_features: Optional[list[str]] = None,
      categorical_features: Optional[list[str]] = None,
  ) -> Dict[str, pd.DataFrame]:
    """Pre-process textual features."""

    if string_features is None:
      string_features = self.string_features
    if categorical_features is None:
      categorical_features = self.categorical_features
    textual_features = string_features + categorical_features
    text_cleaning_func = functools.partial(
        preprocessing.clean_text,
        lowercase=self.lowercase,
        remove_punctuation=self.remove_punctuation,
        remove_https=self.remove_https,
        remove_html=self.remove_html,
        remove_non_alphanumeric=self.remove_non_alphanumeric,
        truncate_len=self.truncate_len,
    )

    # 1. Handle missing values
    if len(string_features) > 0:  # pylint: disable=g-explicit-length-test
      data = preprocessing.handle_text_features_with_missing_values(
          data=data,
          feature_keys=string_features,
          nan_policy=self.string_nan_policy,
      )
    if len(categorical_features) > 0:  # pylint: disable=g-explicit-length-test
      data = preprocessing.handle_text_features_with_missing_values(
          data=data,
          feature_keys=categorical_features,
          nan_policy=self.categorical_nan_policy,
      )

    # 2. Clean up textual features
    data["train"].loc[:, textual_features] = (
        data["train"].loc[:, textual_features].map(text_cleaning_func)
    )
    data["val"].loc[:, textual_features] = (
        data["val"].loc[:, textual_features].map(text_cleaning_func)
    )
    data["test"].loc[:, textual_features] = (
        data["test"].loc[:, textual_features].map(text_cleaning_func)
    )

    return data
