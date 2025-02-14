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

from absl.testing import absltest
from ingestables.torch import types
from ingestables.torch.model import head
import torch
from torch import nn


class IdentityAligner(nn.Module):

  def forward(
      self,
      x_keys: torch.Tensor,
      x_vals: torch.Tensor,
  ) -> torch.Tensor:
    return x_keys, x_vals


class IdentityKvCombiner(nn.Module):

  def forward(
      self,
      z_key_emb: torch.Tensor,
      z_val_emb: torch.Tensor,
  ) -> torch.Tensor:
    del z_key_emb
    return z_val_emb


class ClassificationTest(absltest.TestCase):

  def test_classification_one_feature(self):
    max_num_classes = 2
    z_emb = torch.as_tensor(
        [
            [[1, 1, 1]],
            [[-1, -1, -1]],
            [[1, 1, 1]],
            [[-1, -1, -1]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 1, 3]
    x_keys = torch.zeros(
        4,
        1,
        3,
        dtype=torch.float32,
    )  # shape: [4, 1, 3]
    x_vals_all = torch.as_tensor(
        [
            [[[-1, -1, -1], [1, 1, 1]]],
            [[[-1, -1, -1], [1, 1, 1]]],
            [[[-1, -1, -1], [1, 1, 1]]],
            [[[-1, -1, -1], [1, 1, 1]]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 1, 2, 3]
    padding = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    mask = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    missing = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    expected_logits = torch.as_tensor(
        [
            [[-3, 3]],
            [[3, -3]],
            [[-3, 3]],
            [[3, -3]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 1, 2]

    aligner = IdentityAligner()
    kv_combiner = IdentityKvCombiner()
    classification_head = head.IngesTablesClassification(
        aligner=aligner,
        kv_combiner=kv_combiner,
        max_num_classes=max_num_classes,
    )
    inference_inputs = types.IngesTablesInferenceInputs(
        x_keys=x_keys,
        x_vals=x_vals_all,
        mask=mask,
        missing=missing,
        x_vals_all=x_vals_all,
        padding=padding,
    )
    actual_logits = classification_head(
        z_emb,
        inference_inputs=inference_inputs,
    )  # shape: [4, 1, 2]

    self.assertEqual(actual_logits.shape, (4, 1, 2))
    self.assertTrue(torch.equal(expected_logits, actual_logits))

  def test_classification_two_features(self):
    max_num_classes = 2
    z_emb = torch.as_tensor(
        [
            [[1, 1, 1], [-1, -1, -1]],
            [[-1, -1, -1], [1, 1, 1]],
            [[1, 1, 1], [-1, -1, -1]],
            [[-1, -1, -1], [1, 1, 1]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 2, 3]
    x_keys = torch.zeros(
        4,
        2,
        3,
        dtype=torch.float32,
    )  # shape: [4, 2, 3]
    x_vals_all = torch.as_tensor(
        [
            [[[-1, -1, -1], [1, 1, 1]], [[-1, -1, -1], [1, 1, 1]]],
            [[[-1, -1, -1], [1, 1, 1]], [[-1, -1, -1], [1, 1, 1]]],
            [[[-1, -1, -1], [1, 1, 1]], [[-1, -1, -1], [1, 1, 1]]],
            [[[-1, -1, -1], [1, 1, 1]], [[-1, -1, -1], [1, 1, 1]]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 2, 2, 3]
    padding = torch.ones(
        4,
        2,
        2,
        dtype=torch.bool,
    )  # shape: [4, 2, 2]
    mask = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    missing = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    expected_logits = torch.as_tensor(
        [
            [[-3, 3], [3, -3]],
            [[3, -3], [-3, 3]],
            [[-3, 3], [3, -3]],
            [[3, -3], [-3, 3]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 2, 2]

    aligner = IdentityAligner()
    kv_combiner = IdentityKvCombiner()
    classification_head = head.IngesTablesClassification(
        aligner=aligner,
        kv_combiner=kv_combiner,
        max_num_classes=max_num_classes,
    )
    inference_inputs = types.IngesTablesInferenceInputs(
        x_keys=x_keys,
        x_vals=x_vals_all,
        mask=mask,
        missing=missing,
        x_vals_all=x_vals_all,
        padding=padding,
    )
    actual_logits = classification_head(
        z_emb,
        inference_inputs=inference_inputs,
    )  # shape: [4, 2, 2]

    self.assertEqual(actual_logits.shape, (4, 2, 2))
    self.assertTrue(torch.equal(expected_logits, actual_logits))

  def test_classification_two_features_diff_cardinality(self):
    # In this scenario, the first feature has 3 classes, while the second
    # feature has 2 classes. We want to make sure that the classification head
    # produces the expected output when given the correct padding.
    max_num_classes = 3
    z_emb = torch.as_tensor(
        [
            [[1, 1, 1], [-1, -1, -1]],
            [[-1, -1, -1], [1, 1, 1]],
            [[1, 1, 1], [-1, -1, -1]],
            [[-1, -1, -1], [1, 1, 1]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 2, 3]
    x_keys = torch.zeros(
        4,
        2,
        3,
        dtype=torch.float32,
    )  # shape: [4, 2, 3]
    x_vals_all = torch.as_tensor(
        [
            [
                [[-1, -1, -1], [1, 1, 1], [1, -2, 1]],
                [[-1, -1, -1], [1, 1, 1], [0, 0, 0]],
            ],
            [
                [[-1, -1, -1], [1, 1, 1], [1, -2, 1]],
                [[-1, -1, -1], [1, 1, 1], [0, 0, 0]],
            ],
            [
                [[-1, -1, -1], [1, 1, 1], [1, -2, 1]],
                [[-1, -1, -1], [1, 1, 1], [0, 0, 0]],
            ],
            [
                [[-1, -1, -1], [1, 1, 1], [1, -2, 1]],
                [[-1, -1, -1], [1, 1, 1], [0, 0, 0]],
            ],
        ],
        dtype=torch.float32,
    )  # shape: [4, 2, 3, 3]
    padding = torch.as_tensor(
        [
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
        ],
        dtype=torch.bool,
    )  # shape: [4, 2, 3]
    mask = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    missing = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    expected_logits = torch.as_tensor(
        [
            [[-3, 3, 0], [3, -3, float("-inf")]],
            [[3, -3, 0], [-3, 3, float("-inf")]],
            [[-3, 3, 0], [3, -3, float("-inf")]],
            [[3, -3, 0], [-3, 3, float("-inf")]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 2, 3]

    aligner = IdentityAligner()
    kv_combiner = IdentityKvCombiner()
    classification_head = head.IngesTablesClassification(
        aligner=aligner,
        kv_combiner=kv_combiner,
        max_num_classes=max_num_classes,
    )
    inference_inputs = types.IngesTablesInferenceInputs(
        x_keys=x_keys,
        mask=mask,
        missing=missing,
        x_vals_all=x_vals_all,
        x_vals=x_vals_all,
        padding=padding,
    )
    actual_logits = classification_head(
        z_emb, inference_inputs=inference_inputs
    )  # shape: [4, 2, 3]

    self.assertEqual(actual_logits.shape, (4, 2, 3))
    self.assertTrue(torch.equal(expected_logits, actual_logits))


class RegressionTest(absltest.TestCase):

  def test_regression_two_features(self):
    z_emb = torch.as_tensor(
        [
            [[1, 1, 1], [-1, -1, -1]],
            [[-1, -1, -1], [1, 1, 1]],
            [[1, 1, 1], [-1, -1, -1]],
            [[-1, -1, -1], [1, 1, 1]],
        ],
        dtype=torch.float32,
    )  # shape: [4, 2, 3]
    padding = torch.as_tensor(
        [
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
        ],
        dtype=torch.bool,
    )  # shape: [4, 2, 3]
    mask = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]
    missing = torch.ones(
        4,
        1,
        2,
        dtype=torch.bool,
    )  # shape: [4, 1, 2]

    regression_head = head.IngesTablesRegression(z_dim=3)
    logits = regression_head(
        z_emb,
        inference_inputs=types.IngesTablesInferenceInputs(
            x_keys=z_emb,
            x_vals=z_emb,
            mask=mask,
            missing=missing,
            padding=padding,
        ),
    )  # shape: [4, 2, 1]

    self.assertEqual(logits.shape, (4, 2, 1))


if __name__ == "__main__":
  absltest.main()
