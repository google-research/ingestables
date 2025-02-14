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

"""Encoders."""

import copy
import dataclasses
import logging
from typing import Dict, List, Literal, Optional, Tuple
import warnings

from etils import etree
from ingestables.torch import types
from ingestables.torch.data import encoding
from ingestables.torch.data import preprocessing
from ingestables.torch.data import serialization
from ingestables.torch.data import utils
from ingestables.torch.model import text_encoders
import numpy as np
import pandas as pd
from sklearn import preprocessing as sklearn_preprocessing
import torch
import torch.nn.functional as F
from torch.utils import data as torch_data
import tqdm


@dataclasses.dataclass
class EncodedFeatureNames:
  """This is a dataclass that holds encoded feature names for IngesTables models.

  Here text_emb_dim refers to the embedding dimension of the text encoder.

  Attributes:
    categorical: Encoded categorical feature names. Shape: [num_cat_feats,
      text_emb_dim]. None if there are no categorical features.
    numeric: Encoded numeric feature names. Shape: [num_num_feats,
      text_emb_dim].
    string: Encoded string feature names. Shape: [num_str_feats, text_emb_dim].
  """

  categorical: Optional[torch.Tensor] = None
  numeric: Optional[torch.Tensor] = None
  string: Optional[torch.Tensor] = None


@dataclasses.dataclass
class PreprocessedInputs:
  """This is a dataclass that holds preprocessed inputs for different models.

  This dataclass holds all the necessary information for training and evaluating
  all implemented tabular models.

  In the following description, `text_emb_dim` refers to the embedding dimension
  of the text encoder.

  Attributes:
    feature_type_dict: Dict[str, List[str]] The values of this dictionary are
      the names of features of each type (categorical, numeric, string).
    encoded_feature_names: An EncodedFeatureNames object which holds encoded
      feature names of each type (categorical, numeric, string).
    encoded_targets: The encoded targets of shape [num_examples, num_targets].
    raw_numeric: Raw numeric features of shape [num_examples,
      num_numeric_feats].
    encoded_numeric: Encoded numeric features of shape [num_examples,
      num_numeric_feats, num_bins]. num_bins = 1 for raw encoding.
    encoded_categorical: Encoded categorical features of shape [num_examples,
      num_categorical_feats, text_emb_dim].
    encoded_categorical_ordinal: Ordinal encoding of categorical features. This
      is used in IngesTables models. Shape: [num_examples,
      num_categorical_feats, 1].
    categorical_value_embeddings: Embeddings for all unique values of each
      categorical feature, of shape [n_cat_features, max_num_categories,
      text_emb_dim].
    categorical_value_padding: Padding for categorical features. Shape:
      [n_cat_features, max_num_categories].
    encoded_string: Encoded string features of shape [num_examples,
      num_string_feats, text_emb_dim].
  """

  # TODO(mononito): Add support for Jaxtyping or torchtyping

  # Feature names and types
  feature_type_dict: Dict[str, List[str]]
  encoded_feature_names: EncodedFeatureNames
  # Targets
  encoded_targets: torch.Tensor | None = None
  # Numeric features
  raw_numeric: torch.Tensor | None = None
  encoded_numeric: torch.Tensor | None = None
  # Categorical features
  encoded_categorical: torch.Tensor | None = None
  encoded_categorical_ordinal: torch.Tensor | None = None
  categorical_value_embeddings: torch.Tensor | None = None
  categorical_value_padding: torch.Tensor | None = None
  # String features
  encoded_string: torch.Tensor = None


class Encoder:
  """Encoder class.

  This class is responsible for encoding the data. As such it does the
  following:
  1. Infers feature types (categorical, numeric, string).
  2. Encodes different types of features.
  3. Rearranges the features such that the target feature is always the first
     one.
  """

  def __init__(
      self,
      max_num_categories: int = 8,
      n_bins: int = 128,
      binning_method: Literal["target-aware", "quantile", "uniform"] = (
          "uniform"
      ),
      target_encoding: Literal["raw", "label_encoding", "llm"] = "raw",
      numeric_encoding: Literal["raw", "soft_one_hot", "piecewise_linear"] = (
          "piecewise_linear"
      ),
      categorical_encoding: Literal["raw", "one_hot", "ordinal", "llm"] = "llm",
      string_encoding: Literal["raw", "llm", "none"] = "llm",
      feature_name_encoding: Literal["raw", "llm", "none", "ones"] = "llm",
      batch_size: int = 1024,
      text_encoder: text_encoders.TextEncoder | None = None,
      remove_target_from_feature: bool = False,
      serialize_columns: bool = False,
      feature_serializer: serialization.FeatureSerializer | None = None,
  ):
    """Initialize the encoder.

    When string_encoding and feature_name_encoding is "none", string features
    are excluded from the PreprocessedInputs, and feature names are not encoded.

    Args:
      max_num_categories: The maximum number of categories for categorical
        features.
      n_bins: The number of bins for numeric features.
      binning_method: The binning method for numeric features.
      target_encoding: The target encoding method.
      numeric_encoding: The numeric encoding method.
      categorical_encoding: The categorical encoding method.
      string_encoding: The string encoding method. If "none", string features
        are excluded from the PreprocessedInputs.
      feature_name_encoding: The feature name encoding method.
      batch_size: The batch size.
      text_encoder: The text encoder.
      remove_target_from_feature: Whether to remove the target from the feature.
      serialize_columns: Whether to serialize the columns, for TabLLM baselines.
      feature_serializer: Serialize the all columns except the target_key.

    Raises:
      ValueError: If the text encoder is not provided.
    """
    llm_used = "llm" in [
        categorical_encoding,
        string_encoding,
        feature_name_encoding,
        target_encoding,
    ]
    if llm_used and not text_encoder:
      raise ValueError("Text encoder is required if it is used for encoding.")
    self.max_num_categories = max_num_categories
    self.n_bins = n_bins
    self.binning_method = binning_method
    self.target_encoding = target_encoding
    self.numeric_encoding = numeric_encoding
    self.categorical_encoding = categorical_encoding
    self.batch_size = batch_size
    self.string_encoding = string_encoding
    self.feature_name_encoding = feature_name_encoding
    self.text_encoder = text_encoder
    self.remove_target_from_feature = remove_target_from_feature
    # "none" when text encoder is not used. This is generally the case for
    # non-IngesTables models.
    self.text_encoder_name = (
        "none" if text_encoder is None else text_encoder.text_encoder_name
    )
    self.text_encoder_n_dims = (
        0 if text_encoder is None else text_encoder.embedding_dim
    )
    if serialize_columns and feature_serializer is None:
      raise ValueError(
          "Feature serializer is required if serialize_columns is True."
      )
    self.serialize_columns = serialize_columns
    self.feature_serializer = feature_serializer

  def __call__(
      self,
      data: Dict[str, pd.DataFrame],
      task_info: types.SupervisedTaskInfo,
  ) -> Dict[str, PreprocessedInputs]:
    """Encode a dictionary of dataframes."""

    logging.info("Encoding dataset %s...", task_info.dataset_name)

    # Serializes the all columns except the target_key.
    # Replaces the df's for all splits in-place.
    # Old df: [feat1, feat2, feat3, target]
    # New df: [ textualized_features_string, target] where
    # textualized_features_string is typically something like
    # "feat1 is val1. feat2 is val2. feat3 is val3". The exact serialization
    # strategy can be overridden by modifying the feature_serializer.
    if self.feature_serializer is not None:
      data = self.feature_serializer(data, task_info)

    # Infer feature types
    _, feature_type_dict = utils.infer_feature_types(
        data["train"], self.max_num_categories
    )
    self._check_unimplemented_cases(
        data=data, task_info=task_info, feature_type_dict=feature_type_dict
    )

    # Sort features by feature name
    self.numeric_features, self.categorical_features, self.string_features = (
        self._sort_features(feature_type_dict, task_info)
    )
    feature_type_dict["numeric"] = self.numeric_features
    feature_type_dict["categorical"] = self.categorical_features
    feature_type_dict["string"] = self.string_features

    # Encode feature names
    encoded_feature_names = None
    if self.feature_name_encoding != "none":
      encoded_feature_names = self._encode_feature_names(feature_type_dict)

    # Encode targets for non-IngesTables models
    encoded_targets = self._encode_targets(
        train_targets=data["train"][task_info.target_key].to_numpy(),
        val_targets=data["val"][task_info.target_key].to_numpy(),
        test_targets=data["test"][task_info.target_key].to_numpy(),
        task_info=task_info,
    )

    if self.remove_target_from_feature:
      # When feature serializer is not None, the target_key is removed in the
      # feature serializer.
      if task_info.task_type == "classification":
        feature_type_dict["categorical"].remove(task_info.target_key)
      elif task_info.task_type == "regression":
        feature_type_dict["numeric"].remove(task_info.target_key)

    # Encode numeric features
    raw_numeric = {"train": None, "val": None, "test": None}
    encoded_numeric = {"train": None, "val": None, "test": None}
    if len(self.numeric_features) > 0:  # pylint: disable=g-explicit-length-test
      raw_numeric, encoded_numeric = self._encode_numeric_features(
          num_vals_train=data["train"].loc[:, self.numeric_features].to_numpy(),
          num_vals_val=data["val"].loc[:, self.numeric_features].to_numpy(),
          num_vals_test=data["test"].loc[:, self.numeric_features].to_numpy(),
          targets=data["train"].loc[:, task_info.target_key].to_numpy(),
          task_info=task_info,
      )

    # Encode categorical features
    encoded_categorical = {"train": None, "val": None, "test": None}
    categorical_value_embeddings = None
    categorical_value_padding = None
    encoded_categorical_ordinal = {"train": None, "val": None, "test": None}

    if len(self.categorical_features) > 0:  # pylint: disable=g-explicit-length-test
      (
          encoded_categorical,
          encoded_categorical_ordinal,
          categorical_value_embeddings,
          categorical_value_padding,
      ) = self._encode_categorical_features(
          train_cat_vals=data["train"][self.categorical_features].to_numpy(),
          val_cat_vals=data["val"][self.categorical_features].to_numpy(),
          test_cat_vals=data["test"][self.categorical_features].to_numpy(),
      )

    # Encode string features
    encoded_string = {"train": None, "val": None, "test": None}
    if len(self.string_features) > 0 and self.string_encoding != "none":  # pylint: disable=g-explicit-length-test
      encoded_string = self._encode_string_features(
          train_str_vals=data["train"][self.string_features].to_numpy(),
          val_str_vals=data["val"][self.string_features].to_numpy(),
          test_str_vals=data["test"][self.string_features].to_numpy(),
      )

    encoded_inputs = {
        "train": PreprocessedInputs(
            feature_type_dict=feature_type_dict,
            encoded_feature_names=encoded_feature_names,
            encoded_targets=encoded_targets["train"],
            raw_numeric=raw_numeric["train"],
            encoded_numeric=encoded_numeric["train"],
            encoded_categorical=encoded_categorical["train"],
            encoded_categorical_ordinal=encoded_categorical_ordinal["train"],
            categorical_value_embeddings=categorical_value_embeddings,
            categorical_value_padding=categorical_value_padding,
            encoded_string=encoded_string["train"],
        ),
        "val": PreprocessedInputs(
            feature_type_dict=feature_type_dict,
            encoded_feature_names=encoded_feature_names,
            encoded_targets=encoded_targets["val"],
            raw_numeric=raw_numeric["val"],
            encoded_numeric=encoded_numeric["val"],
            encoded_categorical=encoded_categorical["val"],
            encoded_categorical_ordinal=encoded_categorical_ordinal["val"],
            categorical_value_embeddings=categorical_value_embeddings,
            categorical_value_padding=categorical_value_padding,
            encoded_string=encoded_string["val"],
        ),
        "test": PreprocessedInputs(
            feature_type_dict=feature_type_dict,
            encoded_feature_names=encoded_feature_names,
            encoded_targets=encoded_targets["test"],
            raw_numeric=raw_numeric["test"],
            encoded_numeric=encoded_numeric["test"],
            encoded_categorical=encoded_categorical["test"],
            encoded_categorical_ordinal=encoded_categorical_ordinal["test"],
            categorical_value_embeddings=categorical_value_embeddings,
            categorical_value_padding=categorical_value_padding,
            encoded_string=encoded_string["test"],
        ),
    }

    return encoded_inputs

  def __repr__(self):
    return (
        f"Encoder(max_num_categories={self.max_num_categories},"
        + f" n_bins={self.n_bins}, binning_method={self.binning_method},"
        + f" target_encoding={self.target_encoding},"
        + f" numeric_encoding={self.numeric_encoding},"
        + f" categorical_encoding={self.categorical_encoding},"
        + f" batch_size={self.batch_size}),"
        + f" string_encoding={self.string_encoding},"
        + f" feature_name_encoding={self.feature_name_encoding},"
        + f" text_encoder_name={self.text_encoder_name}"
    )

  def _check_unimplemented_cases(
      self,
      data: Dict[str, pd.DataFrame],
      task_info: types.SupervisedTaskInfo,
      feature_type_dict: Dict[str, List[str]],
  ):
    """Check for unimplemented cases.

    Args:
      data: A dictionary of dataframes.
      task_info: Information about the task.
      feature_type_dict: A dictionary of feature types.

    Raises:
      NotImplementedError: If the number of features in the train and test
        splits are not the same, or if the target key is not present in the
        train split, or if the number of train and test classes are not the
        same.
    """

    # Validation split is always in-distribution
    n_train_features = data["train"].shape[1]
    n_test_features = data["test"].shape[1]

    if n_train_features != n_test_features:
      raise NotImplementedError(
          "The number of features in the train and test splits are"
          + " not the same."
      )

    if task_info.target_key not in data["test"].columns:
      raise NotImplementedError(
          "The target key is not present in the train split."
      )

    if task_info.task_type == "classification":
      train_classes = data["train"][task_info.target_key].unique()
      test_classes = data["test"][task_info.target_key].unique()
      n_train_classes = len(train_classes)
      n_test_classes = len(test_classes)
      if n_train_classes != n_test_classes:
        raise NotImplementedError(
            "The number of train and test classes are not the same"
            f" {n_train_classes} vs. {n_test_classes} \n"
            f" {train_classes}\nvs.\n{test_classes}"
        )

      categorical_features = feature_type_dict["categorical"]
      for cat_feature in categorical_features:
        n_train_cat_vals = len(data["train"][cat_feature].unique())
        n_test_cat_vals = len(data["test"][cat_feature].unique())
        if n_train_cat_vals != n_test_cat_vals:
          # [NOTE] We can handle different number of entities.
          # TODO(mononito): Improve handling for IngesTables.
          warnings.warn(
              "Train and test sets have different number of entities: %s"
          )

  def _sort_features(
      self,
      feature_type_dict: Dict[str, List[str]],
      task_info: types.SupervisedTaskInfo,
  ) -> Tuple[List[str], List[str], List[str]]:
    """Sort features.

    Sorts features such that all feature names are sorted in alphabetical order
    and the target feature is always the first one.

    Args:
      feature_type_dict: A dictionary of feature types.
      task_info: Information about the task.

    Returns:
      A tuple of lists of numeric, categorical, and string features.
    """

    numeric_features = sorted(feature_type_dict["numeric"])
    categorical_features = sorted(feature_type_dict["categorical"])
    string_features = sorted(feature_type_dict["string"])

    target_name = task_info.target_key
    # Make sure the target feature is always the first one.
    if target_name in numeric_features:
      numeric_features.remove(target_name)
      numeric_features.insert(0, target_name)
    else:
      categorical_features.remove(target_name)
      categorical_features.insert(0, target_name)

    return numeric_features, categorical_features, string_features

  def _encode_feature_names(
      self,
      feature_type_dict: Dict[str, List[str]],
  ) -> EncodedFeatureNames:
    """Encode feature names.

    Args:
      feature_type_dict: A dictionary of feature types.

    Returns:
      An EncodedFeatureNames object with the feature names encoded.
    """

    if self.feature_name_encoding == "ones":
      text_encoder = text_encoders.TextEncoder(
          encoding_batch_size=1024,
          text_encoder_name="stub",
          embedding_dim=self.text_encoder_n_dims,
      )
    else:
      text_encoder = self.text_encoder

    numeric, categorical, string = None, None, None
    if len(feature_type_dict["numeric"]) > 0:  # pylint: disable=g-explicit-length-test
      numeric = text_encoder(feature_type_dict["numeric"])
    if len(feature_type_dict["categorical"]) > 0:  # pylint: disable=g-explicit-length-test
      categorical = text_encoder(feature_type_dict["categorical"])
    if len(feature_type_dict["string"]) > 0:  # pylint: disable=g-explicit-length-test
      string = text_encoder(feature_type_dict["string"])

    return EncodedFeatureNames(
        categorical=categorical,
        numeric=numeric,
        string=string,
    )

  def _encode_targets(
      self,
      train_targets: np.ndarray,
      val_targets: np.ndarray,
      test_targets: np.ndarray,
      task_info: types.TaskInfo,
  ) -> Dict[str, torch.Tensor]:
    """Encode the target variable.

    Args:
      train_targets: The training targets.
      val_targets: The validation targets.
      test_targets: The test targets.
      task_info: Information about the task.

    Returns:
      A dictionary of encoded targets.
    """
    encoded_targets = {"train": None, "val": None, "test": None}

    if (
        self.target_encoding == "raw"
        and task_info.task_type == "classification"
    ):
      warnings.warn(
          "Raw target encoding is not supported for classification tasks."
          " Setting to `label_encoding`."
      )
      # [NOTE] For IngesTables, we handle target encoding in another way.
      self.target_encoding = "label_encoding"
    if (
        self.target_encoding == "label_encoding"
        and task_info.task_type == "regression"
    ):
      warnings.warn(
          "Label encoding is not supported for regression tasks. Setting to"
          " `raw`."
      )
      self.target_encoding = "raw"

    if self.target_encoding == "raw":
      encoded_targets = {
          "train": torch.from_numpy(train_targets).float(),
          "val": torch.from_numpy(val_targets).float(),
          "test": torch.from_numpy(test_targets).float(),
      }
    elif self.target_encoding in ["llm", "label_encoding"]:
      label_encoder = sklearn_preprocessing.LabelEncoder()

      encoded_targets = {
          "train": label_encoder.fit_transform(train_targets),
          "val": label_encoder.transform(val_targets),
          "test": label_encoder.transform(test_targets),
      }

      encoded_targets = etree.map(
          lambda x: torch.from_numpy(x).long(), encoded_targets
      )

      if self.target_encoding == "llm":
        target_embs = self.text_encoder(label_encoder.classes_.tolist())
        # (# targets, text_emb_dim)
        encoded_targets["train"] = etree.map(
            lambda x: target_embs[x, :], encoded_targets["train"]
        )
        encoded_targets["val"] = etree.map(
            lambda x: target_embs[x, :], encoded_targets["val"]
        )
        encoded_targets["test"] = etree.map(
            lambda x: target_embs[x, :], encoded_targets["test"]
        )

    return encoded_targets

  def _encode_numeric_features(
      self,
      num_vals_train: np.ndarray,
      num_vals_val: np.ndarray,
      num_vals_test: np.ndarray,
      targets: np.ndarray,
      task_info: types.TaskInfo,
  ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Encode numeric features.

    Args:
      num_vals_train: The training numeric feature values.
      num_vals_val: The validation numeric feature values.
      num_vals_test: The test numeric feature values.
      targets: The targets.
      task_info: Information about the task.

    Returns:
      A tuple of dictionaries, where the first dictionary contains the raw
      numeric feature values and the second dictionary contains the encoded
      numeric feature values.
    """

    encoded_numeric = {}
    raw_numeric = {
        "train": num_vals_train,
        "val": num_vals_val,
        "test": num_vals_test,
    }

    # 1. Compute bins
    bin_edges = preprocessing.compute_bins(
        numeric_feature_values_raw=num_vals_train,
        n_bins=self.n_bins,
        binning_method=self.binning_method,
        target_values=targets,
        task=task_info.task_type,
    )
    bin_edges = torch.from_numpy(bin_edges)
    preprocessing.validate_bins(bin_edges)

    make_dataloader = lambda dataset: torch_data.DataLoader(
        dataset,
        batch_size=self.batch_size,
        shuffle=False,
        drop_last=False,
    )
    # Change numpy arrays to torch tensors
    raw_numeric = etree.map(lambda x: torch.from_numpy(x).float(), raw_numeric)

    # 2. Encode numeric features
    if self.numeric_encoding == "raw":
      encoded_numeric = {
          "train": num_vals_train,
          "val": num_vals_val,
          "test": num_vals_test,
      }
      encoded_numeric = etree.map(
          lambda x: torch.from_numpy(x).float().unsqueeze(-1), encoded_numeric
      )
    elif self.numeric_encoding == "soft_one_hot":
      for k, v in raw_numeric.items():  # pylint: disable=attribute-error
        encoded_vals = []
        dataloader = make_dataloader(v)
        for x in tqdm.tqdm(
            dataloader, total=len(dataloader), desc="Numeric encoding:"
        ):
          encoded_vals.append(
              encoding.soft_one_hot_encoding(x=x, edges=bin_edges)
          )
        encoded_numeric[k] = torch.cat(encoded_vals, dim=0)

    elif self.numeric_encoding == "piecewise_linear":
      for k, v in raw_numeric.items():  # pylint: disable=attribute-error
        encoded_vals = []
        dataloader = make_dataloader(v)
        for x in tqdm.tqdm(
            dataloader, total=len(dataloader), desc="Numeric encoding:"
        ):
          encoded_vals.append(
              encoding.piecewise_linear_encoding(x=x, edges=bin_edges)
          )
        encoded_numeric[k] = torch.cat(encoded_vals, dim=0)
    else:
      raise ValueError(
          f"Unsupported numeric encoding method: {self.numeric_encoding=}"
      )

    return raw_numeric, encoded_numeric

  def _encode_categorical_features(
      self,
      train_cat_vals: np.ndarray,
      val_cat_vals: np.ndarray,
      test_cat_vals: np.ndarray,
  ) -> tuple[
      Dict[str, torch.Tensor],
      Dict[str, torch.Tensor],
      Optional[torch.Tensor],
      Optional[torch.Tensor],
  ]:
    """Encode categorical features."""

    # Ordinal encoding for all categorical features
    encoded_categorical_ordinal = {"train": None, "val": None, "test": None}
    # Embeddings for all unique values of each categorical feature
    categorical_value_embeddings = None
    # Padding to densely pack embeddings of all categorical features together
    categorical_value_padding = None

    raw_categorical = {
        "train": train_cat_vals,
        "val": val_cat_vals,
        "test": test_cat_vals,
    }

    if self.categorical_encoding == "raw":
      encoded_categorical = copy.deepcopy(raw_categorical)
    elif self.categorical_encoding in ["ordinal", "llm"]:
      ordinal_encoder = sklearn_preprocessing.OrdinalEncoder(
          handle_unknown="use_encoded_value",
          unknown_value=-1,
      )
      # TODO(mononito): IngesTables can handle previously unseen categorical
      # values in the test set.
      encoded_categorical_ordinal = {
          "train": ordinal_encoder.fit_transform(train_cat_vals),
          "val": ordinal_encoder.transform(val_cat_vals),
          "test": ordinal_encoder.transform(test_cat_vals),
      }
      encoded_categorical_ordinal = etree.map(
          lambda x: torch.from_numpy(x).long(), encoded_categorical_ordinal
      )
      encoded_categorical = copy.deepcopy(encoded_categorical_ordinal)
      # encoded_categorical_ordinal["train"] is of shape:
      # (n_train_examples, n_categorical_features)

      if self.categorical_encoding == "llm":
        n_categorical_features = len(self.categorical_features)
        categorical_value_embeddings = []
        categorical_value_padding = torch.ones(
            (n_categorical_features, self.max_num_categories),
        ).long()
        for k in range(n_categorical_features):
          cat_emb = self.text_encoder(ordinal_encoder.categories_[k].tolist())
          pad_width = self.max_num_categories - len(
              ordinal_encoder.categories_[k]
          )
          categorical_value_embeddings.append(
              F.pad(cat_emb, (0, 0, 0, pad_width))
          )
          categorical_value_padding[
              k, len(ordinal_encoder.categories_[k]) :
          ] = 0
        categorical_value_embeddings = torch.stack(categorical_value_embeddings)
        # categorical_value_embeddings is of shape:
        # (n_categorical_features, max_num_categories, text_emb_dim)

        train_vals_encs, val_vals_encs, test_vals_encs = [], [], []
        for k in range(n_categorical_features):
          # TODO(mononito): Fix this Linting error (unsure how to do it)
          train_vals_encs.append(
              etree.map(
                  lambda i: categorical_value_embeddings[k, i, :],  # pylint: disable=cell-var-from-loop
                  encoded_categorical_ordinal["train"][:, k],
              )
          )
          val_vals_encs.append(
              etree.map(
                  lambda i: categorical_value_embeddings[k, i, :],  # pylint: disable=cell-var-from-loop
                  encoded_categorical_ordinal["val"][:, k],
              )
          )
          test_vals_encs.append(
              etree.map(
                  lambda i: categorical_value_embeddings[k, i, :],  # pylint: disable=cell-var-from-loop
                  encoded_categorical_ordinal["test"][:, k],
              )
          )
        encoded_categorical["train"] = torch.stack(train_vals_encs, dim=1)
        encoded_categorical["val"] = torch.stack(val_vals_encs, dim=1)
        encoded_categorical["test"] = torch.stack(test_vals_encs, dim=1)

    elif self.categorical_encoding == "one_hot":
      one_hot_encoder = sklearn_preprocessing.OneHotEncoder(
          sparse=False,
          handle_unknown="ignore",
      )
      encoded_categorical = {
          "train": one_hot_encoder.fit_transform(train_cat_vals),
          "val": one_hot_encoder.transform(val_cat_vals),
          "test": one_hot_encoder.transform(test_cat_vals),
      }
      encoded_categorical = etree.map(
          lambda x: torch.from_numpy(x).long(), encoded_categorical
      )
    else:
      raise ValueError(
          f"Unsupported categorical encoding: {self.categorical_encoding}"
      )

    return (
        encoded_categorical,
        encoded_categorical_ordinal,
        categorical_value_embeddings,
        categorical_value_padding,
    )

  def _encode_string_features(
      self,
      train_str_vals: np.ndarray,
      val_str_vals: np.ndarray,
      test_str_vals: np.ndarray,
  ) -> Dict[str, torch.Tensor]:
    """Encode string features."""

    raw_string = {
        "train": train_str_vals,
        "val": val_str_vals,
        "test": test_str_vals,
    }

    if self.string_encoding == "raw":
      encoded_string = copy.deepcopy(raw_string)
    elif self.string_encoding == "llm":

      # Dataloader to ember string features
      # make_dataloader = lambda dataset: torch_data.DataLoader(
      #     dataset,
      #     batch_size=self.batch_size,
      #     shuffle=False,
      #     drop_last=False,
      # )

      encoded_string = {"train": {}, "val": {}, "test": {}}

      for split in ["train", "val", "test"]:
        for i in range(len(self.string_features)):
          # encoded_string[split][i] = []
          # dataloader = make_dataloader(
          #     utils.TorchDataWrapper(raw_string[split][:, i])
          # )
          # for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
          #   encoded_string[split][i].append(self.text_encoder(batch))

          # encoded_string[split][i] = torch.cat(encoded_string[split][i], dim=0)  # pylint: disable=line-too-long
          encoded_string[split][i] = self.text_encoder(
              raw_string[split][:, i].tolist()
          )
        encoded_string[split] = torch.cat(
            [v.unsqueeze(1) for v in encoded_string[split].values()],
            dim=1,
        )

    else:
      raise ValueError(f"Unsupported string encoding {self.string_encoding}")

    return encoded_string
