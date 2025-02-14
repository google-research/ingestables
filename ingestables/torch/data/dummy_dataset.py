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

"""Dummy dataset."""

from ingestables.torch import types
import numpy as np
import pandas as pd
import torch

XorDataFrame = pd.DataFrame({
    "x1": np.array([0.0, 0, 1, 1] * 1000),
    "x2": np.array([0.0, 1, 0, 1] * 1000),
    "y": np.array(["0", "1", "1", "0"] * 1000),
})


class XorDataset(torch.utils.data.Dataset):
  """XOR dataset."""

  def __init__(self):
    super().__init__()
    x1_key_emb = [0, 0, 1]
    x2_key_emb = [0, 1, 0]
    y_key_emb = [1, 0, 0]

    one_val_emb = [1, 1, 1]
    zero_val_emb = [-1, -1, -1]
    false_val_emb = [0, 0, 1]
    true_val_emb = [1, 0, 0]

    # Example 0:
    #   1 xor 1 = 0
    #   {"x1": 1.0, "x2": 1.0, "y": "false"}
    #   cat:
    #       key: "y" --> y_key_emb
    #       val: "false" --> false_val_emb
    #   num:
    #       key: "x1" --> x1_key_emb
    #       val: 1.0 --> one_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: 1.0 --> one_val_emb
    inference_inputs_0 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([false_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
            x_vals_all=np.asarray(
                [[false_val_emb, true_val_emb]], dtype=np.float32
            ),
            padding=np.asarray([[1, 1]], dtype=bool),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([one_val_emb, one_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
        ),
    }
    training_inputs_0 = {
        "cat": types.IngesTablesTrainingInputs(
            # label is 0, because it is "false", category zero.
            y_vals=np.asarray([[0]], dtype=np.int64),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other categorical
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_0 = {
        "cat": types.IngesTablesEvaluationInputs(
            # Index 0 of the categorical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }
    # Example 1:
    #   1 xor 0 = 1
    #   {"x1": 1.0, "x2": 0.0, "y": "true"}
    #   cat:
    #       key: "y" --> y_key_emb
    #       val: "true" --> true_val_emb
    #   num:
    #       key: "x1" --> x1_key_emb
    #       val: 1.0 --> one_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: 0.0 --> zero_val_emb
    inference_inputs_1 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([true_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
            x_vals_all=np.asarray(
                [[false_val_emb, true_val_emb]], dtype=np.float32
            ),
            padding=np.asarray([[1, 1]], dtype=bool),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([one_val_emb, zero_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
        ),
    }
    training_inputs_1 = {
        "cat": types.IngesTablesTrainingInputs(
            # label is 1, because it is "true", category one.
            y_vals=np.asarray([[1]], dtype=np.int64),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other categorical
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_1 = {
        "cat": types.IngesTablesEvaluationInputs(
            # Index 0 of the categorical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }
    # Example 2:
    #   0 xor 1 = 1
    #   {"x1": 0.0, "x2": 0.1, "y": "true"}
    #   cat:
    #       key: "y" --> y_key_emb
    #       val: "true" --> true_val_emb
    #   num:
    #       key: "x1" --> x1_key_emb
    #       val: 0.0 --> zero_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: 1.0 --> one_val_emb
    inference_inputs_2 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([true_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
            x_vals_all=np.asarray(
                [[false_val_emb, true_val_emb]], dtype=np.float32
            ),
            padding=np.asarray([[1, 1]], dtype=bool),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([zero_val_emb, one_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
        ),
    }
    training_inputs_2 = {
        "cat": types.IngesTablesTrainingInputs(
            # label is 1, because it is "true", category one.
            y_vals=np.asarray([[1]], dtype=np.int64),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other categorical
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_2 = {
        "cat": types.IngesTablesEvaluationInputs(
            # Index 0 of the categorical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }
    # Example 3:
    #   0 xor 0 = 0
    #   {"x1": 0.0, "x2": 0.0, "y": "false"}
    #   cat:
    #       key: "y" --> y_key_emb
    #       val: "false" --> false_val_emb
    #   num:
    #       key: "x1" --> x1_key_emb
    #       val: 0.0 --> zero_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: 0.0 --> zero_val_emb
    inference_inputs_3 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([false_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
            x_vals_all=np.asarray(
                [[false_val_emb, true_val_emb]], dtype=np.float32
            ),
            padding=np.asarray([[1, 1]], dtype=bool),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([zero_val_emb, zero_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
        ),
    }
    training_inputs_3 = {
        "cat": types.IngesTablesTrainingInputs(
            # label is 0, because it is "false", category zero.
            y_vals=np.asarray([[0]], dtype=np.int64),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other categorical
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_3 = {
        "cat": types.IngesTablesEvaluationInputs(
            # Index 0 of the categorical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }

    self.data = [
        (inference_inputs_0, training_inputs_0, eval_inputs_0),
        (inference_inputs_1, training_inputs_1, eval_inputs_1),
        (inference_inputs_2, training_inputs_2, eval_inputs_2),
        (inference_inputs_3, training_inputs_3, eval_inputs_3),
    ]

  def __len__(self):
    return 4

  def __getitem__(self, idx):
    sample = self.data[idx]
    return sample


AndDataFrame = pd.DataFrame({
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [1, 0, 0, 1],
})


class AndDataset(torch.utils.data.Dataset):
  """AND dataset."""

  def __init__(self):
    super().__init__()
    # Note that these keys are different from those of XorDataset, so that they
    # can be distinguished.
    x1_key_emb = [0, 0, -1]
    x2_key_emb = [0, -1, 0]
    y_key_emb = [-1, 0, 0]

    one_val_emb = [1, 1, 1]
    zero_val_emb = [-1, -1, -1]
    false_val_emb = [0, 0, 1]
    true_val_emb = [1, 0, 0]
    # Example 0:
    #   1 and 1 = 1
    #   {"x1": "true", "x2": "true", "y": 1.0}
    #   cat:
    #       key: "x1" --> x1_key_emb
    #       val: "true" --> true_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: "true" --> true_val_emb
    #   num:
    #       key: "y" --> y_key_emb
    #       val: 1.0 --> one_val_emb
    inference_inputs_0 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([true_val_emb, true_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
            x_vals_all=np.asarray(
                [
                    [false_val_emb, true_val_emb],
                    [false_val_emb, true_val_emb],
                ],
                dtype=np.float32,
            ),
            padding=np.asarray(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=bool,
            ),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([one_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
        ),
    }
    training_inputs_0 = {
        "num": types.IngesTablesTrainingInputs(
            # label is 1.0, the actual (scaled) numerical value.
            y_vals=np.asarray([[1.0]], dtype=np.float32),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other numeric
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_0 = {
        "num": types.IngesTablesEvaluationInputs(
            # Index 0 of the numerical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }
    # Example 1:
    #   1 and 0 = 1
    #   {"x1": "true", "x2": "false", "y": 1.0}
    #   cat:
    #       key: "x1" --> x1_key_emb
    #       val: "true" --> true_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: "false" --> false_val_emb
    #   num:
    #       key: "y" --> y_key_emb
    #       val: 1.0 --> one_val_emb
    inference_inputs_1 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([true_val_emb, false_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
            x_vals_all=np.asarray(
                [
                    [false_val_emb, true_val_emb],
                    [false_val_emb, true_val_emb],
                ],
                dtype=np.float32,
            ),
            padding=np.asarray(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=bool,
            ),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([one_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
        ),
    }
    training_inputs_1 = {
        "num": types.IngesTablesTrainingInputs(
            # label is 1.0, the actual (scaled) numerical value.
            y_vals=np.asarray([[1.0]], dtype=np.float32),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other numeric
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_1 = {
        "num": types.IngesTablesEvaluationInputs(
            # Index 0 of the numerical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }
    # Example 2:
    #   0 and 1 = 1
    #   {"x1": "false", "x2": "true", "y": 1.0}
    #   cat:
    #       key: "x1" --> x1_key_emb
    #       val: "false" --> false_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: "true" --> true_val_emb
    #   num:
    #       key: "y" --> y_key_emb
    #       val: 1.0 --> one_val_emb
    inference_inputs_2 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([false_val_emb, true_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
            x_vals_all=np.asarray(
                [
                    [false_val_emb, true_val_emb],
                    [false_val_emb, true_val_emb],
                ],
                dtype=np.float32,
            ),
            padding=np.asarray(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=bool,
            ),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([one_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
        ),
    }
    training_inputs_2 = {
        "num": types.IngesTablesTrainingInputs(
            # label is 1.0, the actual (scaled) numerical value.
            y_vals=np.asarray([[1.0]], dtype=np.float32),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other numeric
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_2 = {
        "num": types.IngesTablesEvaluationInputs(
            # Index 0 of the numerical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }
    # Example 3:
    #   0 and 0 = 0
    #   {"x1": "false", "x2": "false", "y": 0.0}
    #   cat:
    #       key: "x1" --> x1_key_emb
    #       val: "false" --> false_val_emb
    #       key: "x2" --> x2_key_emb
    #       val: "false" --> false_val_emb
    #   num:
    #       key: "y" --> y_key_emb
    #       val: 0.0 --> zero_val_emb
    inference_inputs_3 = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([x1_key_emb, x2_key_emb], dtype=np.float32),
            x_vals=np.asarray([false_val_emb, false_val_emb], dtype=np.float32),
            mask=np.asarray([[1], [1]], dtype=bool),
            missing=np.asarray([[1], [1]], dtype=bool),
            x_vals_all=np.asarray(
                [
                    [false_val_emb, true_val_emb],
                    [false_val_emb, true_val_emb],
                ],
                dtype=np.float32,
            ),
            padding=np.asarray(
                [
                    [1, 1],
                    [1, 1],
                ],
                dtype=bool,
            ),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=np.asarray([y_key_emb], dtype=np.float32),
            x_vals=np.asarray([zero_val_emb], dtype=np.float32),
            mask=np.asarray([[0]], dtype=bool),
            missing=np.asarray([[1]], dtype=bool),
        ),
    }
    training_inputs_3 = {
        "num": types.IngesTablesTrainingInputs(
            # label is 0.0, the actual (scaled) numerical value.
            y_vals=np.asarray([[0.0]], dtype=np.float32),
            # loss weight is 1 because we want the label loss here to be
            # back-propogated during training. If there are other numeric
            # features, they would have loss weight of 0 for standard supervised
            # learning.
            loss_weights=np.asarray([[1.0]], dtype=np.float32),
        ),
    }
    eval_inputs_3 = {
        "num": types.IngesTablesEvaluationInputs(
            # Index 0 of the numerical features dimension is the target
            # whose metrics we care about.
            target_index=np.asarray([[0]], dtype=np.int64),
        ),
    }

    self.data = [
        (inference_inputs_0, training_inputs_0, eval_inputs_0),
        (inference_inputs_1, training_inputs_1, eval_inputs_1),
        (inference_inputs_2, training_inputs_2, eval_inputs_2),
        (inference_inputs_3, training_inputs_3, eval_inputs_3),
    ]

  def __len__(self):
    return 4

  def __getitem__(self, idx):
    sample = self.data[idx]
    return sample


StocksDataFrame = pd.DataFrame({
    "x": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "y": np.array([
        180.0,
        181.0,
        182.0,
        183.0,
        184.0,
        185.0,
        186.0,
        187.0,
        188.0,
        189.0,
    ]),
})
