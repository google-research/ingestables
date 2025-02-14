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

"""Feature serialization strategies."""

from typing import Dict
from ingestables.torch import types
import pandas as pd


class FeatureSerializer:
  """Serialize the all columns except the target_key."""

  def textualize_features(self, row, cols_to_textualize):
    text = []
    for i, col in enumerate(row.index):
      if col in cols_to_textualize:
        text.append(f"The {col} is {row[i]}.")
    return " ".join(text)

  def serialize_dataframe(self, df, feature_cols, label_col):
    serialized_df = pd.DataFrame()
    serialized_df["serialized_features"] = df.apply(
        self.textualize_features, axis=1, cols_to_textualize=feature_cols
    )
    serialized_df[label_col] = df[label_col]
    return serialized_df

  def __call__(
      self,
      data: Dict[str, pd.DataFrame],
      task_info: types.SupervisedTaskInfo,
  ) -> Dict[str, pd.DataFrame]:

    label_col = task_info.target_key
    # everything except the label column is a feature
    cols_to_textualize = list(data["train"].columns)
    # remove the label column from the feature columns
    cols_to_textualize.remove(label_col)

    return {
        split: self.serialize_dataframe(
            data[split], cols_to_textualize, label_col
        )
        for split in data
    }
