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

"""Base data classes."""

import dataclasses
from typing import Dict, List, Optional, Protocol
from ingestables.torch import types


@dataclasses.dataclass(kw_only=True)
class FeatureDescription(Protocol):
  feature_type: str
  feature_name: str
  feature_description: Optional[str] = None


@dataclasses.dataclass(kw_only=True)
class CategoricalFeatureDescription(FeatureDescription):
  feature_type: str = "categorical"
  num_categories: int
  categories: List[str]


@dataclasses.dataclass(kw_only=True)
class NumericFeatureDescription(FeatureDescription):
  feature_type: str = "numeric"
  max: float
  min: float
  mean: float
  std: float
  median: float
  units: Optional[str] = None


@dataclasses.dataclass(kw_only=True)
class StringFeatureDescription(FeatureDescription):
  feature_type: str = "string"
  max_length: int
  min_length: int
  example_strings: List[str]


@dataclasses.dataclass(kw_only=True)
class DatasetDescription:
  dataset_name: str
  dataset_description: Optional[str] = None
  num_rows: int
  num_features: int
  num_string_features: int
  num_categorical_features: int
  num_numeric_features: int
  task_information: types.TaskInfo
  feature_descriptions: Dict[str, FeatureDescription]
  feature_keys_dict: Dict[str, List[str]]
