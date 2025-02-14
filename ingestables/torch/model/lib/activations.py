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

"""Activation functions."""

import copy
from torch import nn
from transformers import activations

ACT2CLS = copy.deepcopy(activations.ACT2CLS)
ACT2CLS["selu"] = nn.SELU
ACT2FN = activations.ClassInstantier(ACT2CLS)


def get_activation_fn(activation_string):
  """Get activation function by name.

  Args:
    activation_string: Name of the activation function.

  Returns:
    The activation function.

  Raises:
    KeyError: If the activation function is not found in the ACT2FN mapping.
  """
  if activation_string in ACT2FN:
    return ACT2FN[activation_string]
  else:
    raise KeyError(
        f"function {activation_string} not found in ACT2FN mapping"
        + f" {list(ACT2FN.keys())}"
    )
