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

"""Dataset class for CARTE datasets."""

import functools
import itertools
from typing import Dict, Literal, Optional, Tuple
from ingestables.torch.data import base
from ingestables.torch.data import data_utils
from ingestables.torch.model import text_encoders
import ml_collections
import pandas as pd
import sklearn.preprocessing as sklearn_preprocessing
import torch
from torch.utils import data as torch_data
import tqdm


def get_data_config() -> ml_collections.ConfigDict:
  """Dataset config."""
  config = ml_collections.ConfigDict()

  config.dataset_name = "wine_dot_com_prices"
  config.random_seed = 42
  # Train, val, and test splits
  config.train_ratio = 0.8
  config.val_ratio = 0.1
  config.test_ratio = 0.1
  # Pre-processing
  # Text (String and Categorical features)
  config.do_drop_cols_with_missing_values = True
  config.do_drop_cols_with_one_unique_value = True
  config.lowercase = False
  config.remove_punctuation = True
  config.remove_https = True
  config.remove_html = True
  config.remove_non_alphanumeric = True
  config.truncate_len = 100
  config.string_nan_policy = "default_statement"
  config.categorical_nan_policy = "most_frequent"
  config.max_num_categories = 32
  # Numeric
  config.numeric_nan_policy = "mean"
  config.fill_value = 0
  config.noise = 1e-3
  # Numeric scaling
  config.numeric_scaling_method = "quantile"
  config.num_quantiles = 48
  # Numeric binning
  config.n_bins = 128
  config.binning_method = "quantile"
  # Numeric encoding
  config.encoding_method = "piecewise_linear"
  # Text encoder config
  config.text_enc_cfg = text_encoders.get_text_encoder_config()
  config.text_enc_cfg.batch_size = 1024
  # Masking
  return config


class TabularCarteDataset:
  """Dataset class to load and pre-process CARTE datasets."""

  def __init__(
      self,
      config: ml_collections.ConfigDict,
      text_encoder: Optional[text_encoders.TextEncoder] = None,
  ):
    super().__init__()
    self.dataset_name = config.dataset_name
    self.config = config
    self.random_seed = config.random_seed

    data, self.data_description = self._load_data()
    self.split_data = data_utils.split_dataset(
        data=data,
        task_info=self.data_description.task_information,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_state=config.random_seed,
    )
    if text_encoder is None:
      self.text_encoder = text_encoders.TextEncoder(config.text_enc_cfg)
    else:
      self.text_encoder = text_encoder
    self.text_emb_dim = self.text_encoder.encoder.encoder.embedding_dim
    self.feature_keys_dict = self.data_description.feature_keys_dict
    self.task_information = self.data_description.task_information

    self.text_cleaning_func = functools.partial(
        data_utils.clean_text,
        lowercase=self.config.lowercase,
        remove_punctuation=self.config.remove_punctuation,
        remove_https=self.config.remove_https,
        remove_html=self.config.remove_html,
        remove_non_alphanumeric=self.config.remove_non_alphanumeric,
        truncate_len=self.config.truncate_len,
    )

    self.num_feature_keys = sorted(self.feature_keys_dict["numeric"])
    self.str_feature_keys = sorted(self.feature_keys_dict["string"])
    self.cat_feature_keys = sorted(self.feature_keys_dict["categorical"])
    # TODO(mononito): Handle the case where there are no str, cat, or num feats

    # Preprocess datasets and compute embeddings
    self.compute_num_val_embs()
    if len(self.cat_feature_keys + self.str_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      self.compute_text_val_embs()
    if len(self.cat_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      self.compute_all_cat_feat_val_embs()

    self.compute_feat_key_embs()
    self.compute_target_idx()

    # Note: Check for side effects. For example, does dropping rows, reduce the
    # number of categorical feature / categories?

  def __repr__(self):
    return (
        f"TabularCarteDataset(dataset_name={self.dataset_name},"
        + f" data_description={self.data_description})"
    )

  def _load_data(self) -> Tuple[pd.DataFrame, base.DatasetDescription]:
    return data_utils.get_dataset_and_description(
        dataset_name=self.dataset_name,
        benchmark="carte",
        do_drop_cols_with_missing_values=self.config.do_drop_cols_with_missing_values,
        do_drop_cols_with_one_unique_value=self.config.do_drop_cols_with_one_unique_value,
        max_num_categories=self.config.max_num_categories,
    )

  def compute_num_val_embs(self):
    """Function to preprocess and embed numeric feature values."""

    if len(self.num_feature_keys) == 0:  # pylint: disable=g-explicit-length-test
      # If there are no numeric features
      self.y_vals_num = {"train": None, "val": None, "test": None}
      self.numeric_val_embs = {"train": None, "val": None, "test": None}
      return

    # Otherwise preprocess numeric features
    # 1. Handle missing values
    self.split_data = data_utils.handle_numeric_features_with_missing_values(
        data=self.split_data,
        num_feature_keys=self.num_feature_keys,
        nan_policy=self.config.numeric_nan_policy,
        fill_value=self.config.fill_value,
    )

    # 2. Add noise to numeric features to prevent binning collapse
    self.split_data = data_utils.add_noise_to_numeric_features(
        data=self.split_data,
        num_feature_keys=self.num_feature_keys,
        random_seed=self.random_seed,
        noise=self.config.noise,
    )

    # 3. Scale numeric features
    self.split_data = data_utils.scale_numeric_features(
        data=self.split_data,
        num_feature_keys=self.num_feature_keys,
        scaling_method=self.config.numeric_scaling_method,
        num_quantiles=self.config.num_quantiles,
    )

    # 4. Find bins and encode numeric features
    num_vals_train = (
        self.split_data["train"].loc[:, self.num_feature_keys].to_numpy()
    )
    num_vals_val = (
        self.split_data["val"].loc[:, self.num_feature_keys].to_numpy()
    )
    num_vals_test = (
        self.split_data["test"].loc[:, self.num_feature_keys].to_numpy()
    )
    targets = (
        self.split_data["train"]
        .loc[:, self.task_information.target_key]  # pytype: disable=attribute-error
        .to_numpy()
    )

    self.y_vals_num = {
        "train": num_vals_train,
        "val": num_vals_val,
        "test": num_vals_test,
    }

    # Find bins
    bin_edges = data_utils.compute_bins(
        numeric_feature_values_raw=num_vals_train,
        n_bins=self.config.n_bins,
        binning_method=self.config.binning_method,
        target_values=targets,
        task=self.task_information.task_type,
    )
    bin_edges = torch.from_numpy(bin_edges)
    data_utils.validate_bins(bin_edges)

    # Encode numeric values
    if self.config.encoding_method == "none":
      num_enc_train = torch.from_numpy(num_vals_train).unsqueeze(-1)
      num_enc_val = torch.from_numpy(num_vals_val).unsqueeze(-1)
      num_enc_test = torch.from_numpy(num_vals_test).unsqueeze(-1)

    elif self.config.encoding_method == "soft_one_hot":
      num_enc_train = data_utils.soft_one_hot_encoding(
          x=torch.from_numpy(num_vals_train),
          edges=bin_edges,
      )
      num_enc_val = data_utils.soft_one_hot_encoding(
          x=torch.from_numpy(num_vals_val),
          edges=bin_edges,
      )
      num_enc_test = data_utils.soft_one_hot_encoding(
          x=torch.from_numpy(num_vals_test),
          edges=bin_edges,
      )

    elif self.config.encoding_method == "piecewise_linear":
      num_enc_train = data_utils.piecewise_linear_encoding(
          x=torch.from_numpy(num_vals_train),
          edges=bin_edges,
      )
      num_enc_val = data_utils.piecewise_linear_encoding(
          x=torch.from_numpy(num_vals_val),
          edges=bin_edges,
      )
      num_enc_test = data_utils.piecewise_linear_encoding(
          x=torch.from_numpy(num_vals_test),
          edges=bin_edges,
      )
    else:
      raise ValueError(
          f"Unsupported encoding_method: {self.config.encoding_method=}"
      )

    # num_enc_train[val/test] are tensors of shape:
    # [# rows, # numeric features, encoding dimension]
    # encoding dimension = 1 if encoding is none, and equal to n_bins otherwise

    self.numeric_val_embs = {
        "train": dict(
            zip(self.num_feature_keys, num_enc_train.split(1, dim=1))
        ),
        "val": dict(zip(self.num_feature_keys, num_enc_val.split(1, dim=1))),
        "test": dict(zip(self.num_feature_keys, num_enc_test.split(1, dim=1))),
    }

  def compute_feat_key_embs(self):
    """Function to preprocess and embed all feature keys."""
    feature_keys = (
        self.feature_keys_dict["numeric"]
        + self.feature_keys_dict["categorical"]
        + self.feature_keys_dict["string"]
    )

    # Clean and embed text feature keys
    self.feat_key_embs = dict(
        zip(
            feature_keys,
            self.text_encoder(
                [self.text_cleaning_func(k) for k in feature_keys]
            ).split(1),
        )
    )
    # Dict[str, torch.Tensor], mapping between text feature keys and embeddings

  def compute_all_cat_feat_val_embs(self):
    """Compute embeddings of all possible feature values of all categorical features."""

    if len(self.cat_feature_keys) == 0:  # pylint: disable=g-explicit-length-test
      # If there are no categorical features
      self.x_vals_all = None
      self.padding = None
      return

    self.padding = torch.ones(
        (len(self.cat_feature_keys), self.config.max_num_categories),
        dtype=torch.int,
    )

    x_vals_all = []
    for i, k in enumerate(self.cat_feature_keys):
      # Sort categories. Sklearn label encoder also sorts categories
      cats = sorted(self.data_description.feature_descriptions[k].categories)  # pytype: disable=attribute-error
      cat_embs = self.text_encoder([self.text_cleaning_func(c) for c in cats])

      if len(cats) < self.config.max_num_categories:
        self.padding[i, len(cats) :] = 0
        padding = self.config.max_num_categories - len(cats)
        cat_embs = torch.cat([
            cat_embs,
            torch.zeros((padding, self.text_emb_dim), dtype=torch.float32),
        ])

      assert cat_embs.shape == (
          self.config.max_num_categories,
          self.text_emb_dim,
      )

      x_vals_all.append(cat_embs)

    self.x_vals_all = torch.stack(x_vals_all, dim=0)

  def compute_text_val_embs(self):
    """Function to preprocess and embed text (string & categorical) feature values."""

    # Otherwise, preprocess categorical and string features
    text_feature_keys = self.str_feature_keys + self.cat_feature_keys

    assert self.config.string_nan_policy == "default_statement"

    # 1. Handle missing values
    if len(self.str_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      self.split_data = data_utils.handle_text_features_with_missing_values(
          data=self.split_data,
          feature_keys=self.str_feature_keys,
          nan_policy=self.config.string_nan_policy,
      )
    if len(self.cat_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      self.split_data = data_utils.handle_text_features_with_missing_values(
          data=self.split_data,
          feature_keys=self.cat_feature_keys,
          nan_policy=self.config.categorical_nan_policy,
      )

    # TODO(mononito): Check from here...
    # 2. Clean and trim text features
    train_df = self.split_data["train"].loc[:, text_feature_keys]
    val_df = self.split_data["val"].loc[:, text_feature_keys]
    test_df = self.split_data["test"].loc[:, text_feature_keys]

    train_df = train_df.map(self.text_cleaning_func)
    self.split_data["train"].loc[:, text_feature_keys] = train_df
    self.split_data["val"].loc[:, text_feature_keys] = val_df.map(
        self.text_cleaning_func
    )
    self.split_data["test"].loc[:, text_feature_keys] = test_df.map(
        self.text_cleaning_func
    )

    # Get raw (un-embedded) class values to compute metrics
    if len(self.cat_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      y_vals_cat_train = (
          self.split_data["train"].loc[:, self.cat_feature_keys].to_numpy()
      )
      y_vals_cat_val = (
          self.split_data["val"].loc[:, self.cat_feature_keys].to_numpy()
      )
      y_vals_cat_test = (
          self.split_data["test"].loc[:, self.cat_feature_keys].to_numpy()
      )

      ordinal_encoder = sklearn_preprocessing.OrdinalEncoder().fit(
          y_vals_cat_train
      )
      y_vals_cat_train = ordinal_encoder.transform(y_vals_cat_train)
      y_vals_cat_val = ordinal_encoder.transform(y_vals_cat_val)
      y_vals_cat_test = ordinal_encoder.transform(y_vals_cat_test)

      self.y_vals_cat = {
          "train": y_vals_cat_train,
          "val": y_vals_cat_val,
          "test": y_vals_cat_test,
      }
    else:
      self.y_vals_cat = {"train": None, "val": None, "test": None}
    # NOTE: String values are not reconstruced. y_vals_str is not required.

    # 3. Embed text features
    splits = list(self.split_data.keys())

    make_dataloader = lambda dataset: torch_data.DataLoader(
        dataset,
        batch_size=self.config.text_enc_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    text_feat_datasets = {}
    for feat_key, split in itertools.product(text_feature_keys, splits):
      if split not in text_feat_datasets:
        text_feat_datasets[split] = {}
      text_feat_datasets[split][feat_key] = make_dataloader(
          data_utils.TorchDataWrapper(self.split_data[split].loc[:, feat_key])
      )

    text_val_embs = {}
    pbar_s = tqdm.tqdm(text_feat_datasets, total=len(text_feat_datasets))
    for split in pbar_s:
      pbar_s.set_postfix_str(f"Split: {split}")
      text_val_embs[split] = {}
      for feat_key, dataloader in text_feat_datasets[split].items():
        text_val_embs[split][feat_key] = torch.cat(
            [self.text_encoder(batch) for batch in dataloader], axis=0
        )
    # text_val_embs is a dictionary with Str keys: train, val, test
    # and Dict[str, torch.Tensor] values. The keys of this dictionary are the
    # textual feature keys.

    # Split text_val_embs into string and categorical features
    string_val_embs = {}
    categorical_val_embs = {}
    for split in text_val_embs:
      string_val_embs[split] = {}
      categorical_val_embs[split] = {}
      for feat_key in text_val_embs[split]:
        if feat_key in self.str_feature_keys:
          string_val_embs[split][feat_key] = text_val_embs[split][feat_key]
        else:
          categorical_val_embs[split][feat_key] = text_val_embs[split][feat_key]

    if len(self.str_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      self.string_val_embs = string_val_embs
    else:
      self.string_val_embs = {"train": None, "val": None, "test": None}
    if len(self.cat_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      self.categorical_val_embs = categorical_val_embs
    else:
      self.categorical_val_embs = {"train": None, "val": None, "test": None}

  def compute_target_idx(self):
    """Find index of target feature."""
    task_info = self.task_information
    if self.task_information.task_type == "classification":
      target_idx = self.cat_feature_keys.index(task_info.target_key)  # pytype: disable=attribute-error
      self.target_idx = torch.tensor([target_idx], dtype=torch.int)
    elif self.task_information.task_type == "regression":
      target_idx = self.num_feature_keys.index(task_info.target_key)  # pytype: disable=attribute-error
      self.target_idx = torch.tensor([target_idx], dtype=torch.int)
    else:
      raise ValueError(
          f"Unsupported task_type: {self.task_information.task_type=}"
      )

  # def save(self):
  #   save_base_path = (
  # epath.Path(self.config.save_path) / "CARTE" / self.dataset_name
  # )
  #   save_base_path.mkdir(parents=True, exist_ok=True)

  #   with save_base_path.open("w")

  #  def load(self):
  #    pass


class TorchDataset(torch_data.Dataset):
  """Torch dataset that wraps around TabularCarteDataset."""

  def __init__(
      self,
      dataset: TabularCarteDataset,
      split: Literal["train", "val", "test"] = "train",
  ):
    super().__init__()

    self.str_feature_keys = dataset.str_feature_keys
    self.string_val_embs = dataset.string_val_embs[split]

    self.cat_feature_keys = dataset.cat_feature_keys
    self.categorical_val_embs = dataset.categorical_val_embs[split]
    self.x_vals_all = dataset.x_vals_all
    self.y_vals_cat = dataset.y_vals_cat[split]

    self.num_feature_keys = dataset.num_feature_keys
    self.numeric_val_embs = dataset.numeric_val_embs[split]
    self.y_vals_num = dataset.y_vals_num[split]

    self.feat_key_embs = dataset.feat_key_embs
    self.target_idx = dataset.target_idx
    self.padding = dataset.padding
    self.task_type = dataset.task_information.task_type

  def __getitem__(self, index: int) -> Tuple[
      Dict[str, Dict[str, torch.Tensor]],
      Dict[str, Dict[str, torch.Tensor]],
      Dict[str, Dict[str, torch.Tensor]],
  ]:
    """Getitem implementation.

    Args:
      index: integer index to a dataset example

    Returns:
      inference_inputs: Dict
    """
    inference_inputs = {}
    training_inputs = {}
    eval_inputs = {}

    if len(self.cat_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      categorical_inputs = {
          "x_keys": torch.stack(
              [
                  self.feat_key_embs[feat_key].squeeze()
                  for feat_key in self.cat_feature_keys
              ],
              dim=0,
          ),
          "x_vals": torch.stack(
              [
                  self.categorical_val_embs[feat_key][index].squeeze()
                  for feat_key in self.cat_feature_keys
              ],
              dim=0,
          ),
          "x_vals_all": self.x_vals_all,
          "padding": self.padding,
          # TODO(mononito): Keys are not missing by default, add missingness
          "missing": torch.zeros((len(self.cat_feature_keys), 1)).to(
              torch.bool
          ),
      }
      inference_inputs["cat"] = categorical_inputs

    if len(self.num_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      numeric_inputs = {
          "x_keys": torch.stack(
              [
                  self.feat_key_embs[feat_key].squeeze()
                  for feat_key in self.num_feature_keys
              ],
              dim=0,
          ),
          "x_vals": torch.stack(
              [
                  self.numeric_val_embs[feat_key][index].squeeze()
                  for feat_key in self.num_feature_keys
              ],
              dim=0,
          ),
          # TODO(mononito): Keys are not missing by default, add missingness
          "missing": torch.zeros((len(self.num_feature_keys), 1)).to(
              torch.bool
          ),
      }
      inference_inputs["num"] = numeric_inputs

    if len(self.str_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      string_inputs = {
          "x_keys": torch.stack(
              [
                  self.feat_key_embs[feat_key].squeeze()
                  for feat_key in self.str_feature_keys
              ],
              dim=0,
          ),
          "x_vals": torch.stack(
              [
                  self.string_val_embs[feat_key][index].squeeze()
                  for feat_key in self.str_feature_keys
              ],
              dim=0,
          ),
          # TODO(mononito): Keys are not missing by default, add missingness
          "missing": torch.zeros((len(self.str_feature_keys), 1)).to(
              torch.bool
          ),
      }
      inference_inputs["str"] = string_inputs

    if len(self.num_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      training_inputs["num"] = {}
      training_inputs["num"]["y_vals"] = self.y_vals_num[index].squeeze()
    if len(self.cat_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      training_inputs["cat"] = {}
      training_inputs["cat"]["y_vals"] = self.y_vals_cat[index].squeeze()

    if self.task_type == "classification":
      eval_inputs["cat"] = {}
      eval_inputs["cat"]["target_index"] = self.target_idx
    elif self.task_type == "regression":
      eval_inputs["num"] = {}
      eval_inputs["num"]["target_index"] = self.target_idx
    else:
      raise ValueError(f"Unsupported task_type: {self.task_type=}")

    return inference_inputs, training_inputs, eval_inputs

  def __len__(self) -> int:
    if len(self.cat_feature_keys) > 0:  # pylint: disable=g-explicit-length-test
      return len(self.categorical_val_embs[self.cat_feature_keys[0]])
    else:
      return len(self.numeric_val_embs[self.num_feature_keys[0]])

# Example usage
# data_config = get_data_config()
# tabular_dataset = TabularCarteDataset(data_config, text_encoder=None)
# torch_dataset = TorchDataset(tabular_dataset, split='train')
