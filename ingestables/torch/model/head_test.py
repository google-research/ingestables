# Copyright 2026 The ingestables Authors.
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
from absl.testing import parameterized
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


class ClassificationLossWeightsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("one_feature_masked", 2, 1, [1.0, 0.0]),
      ("one_feature_fractional", 3, 1, [0.25, 0.0, 1.0]),
      ("one_feature_uniform", 3, 1, [1.0, 1.0, 1.0]),
      ("one_feature_zero", 3, 1, [0.0, 0.0, 0.0]),
      ("one_sample", 1, 3, [1.0, 0.0, 0.25]),
      ("one_sample_one_feature", 1, 1, [0.25]),
      ("multiple_features", 2, 3, [1.0, 0.0, 0.5, 0.0, 0.25, 1.0]),
  )
  def test_loss_and_gradients_match_per_example_weights(
      self, batch_size, num_features, weights
  ):
    logits = (
        torch.linspace(
            -2.0, 3.0, batch_size * num_features * 3, dtype=torch.float64
        )
        .reshape(batch_size, num_features, 3)
        .requires_grad_()
    )
    targets = (
        torch.arange(batch_size * num_features).reshape(
            batch_size, num_features, 1
        )
        % 3
    )
    loss_weights = torch.tensor(weights, dtype=logits.dtype).reshape(
        batch_size, num_features, 1
    )
    inputs = types.IngesTablesTrainingInputs(
        y_vals=targets, loss_weights=loss_weights
    )
    classifier = head.IngesTablesClassification(
        IdentityAligner(), IdentityKvCombiner(), max_num_classes=3
    )

    actual = classifier.loss(logits, inputs)
    reference_logits = logits.detach().clone().requires_grad_()
    individual = torch.nn.functional.cross_entropy(
        reference_logits.reshape(-1, 3), targets.reshape(-1), reduction="none"
    ).reshape(batch_size, num_features)
    expected = (individual * loss_weights[..., 0]).mean()

    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, logits)[0]
    expected_grad = torch.autograd.grad(expected, reference_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    self.assertEqual(actual.shape, torch.Size([]))
    torch.testing.assert_close(
        inputs.loss_weights, loss_weights, rtol=0, atol=0
    )

  def test_zero_weight_example_is_unchanged_by_optimizer_step(self):
    embeddings = nn.Parameter(
        torch.tensor([[[0.0, 0.0, 0.0]], [[0.0, 4.0, 1.0]]])
    )
    before = embeddings.detach().clone()
    values = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 1, 3, 3)
    inputs = types.IngesTablesInferenceInputs(
        x_keys=torch.zeros(2, 1, 3),
        x_vals=values,
        x_vals_all=values,
        padding=torch.ones(2, 1, 3, dtype=torch.bool),
        mask=torch.ones(2, 1, 1, dtype=torch.bool),
        missing=torch.zeros(2, 1, 1, dtype=torch.bool),
    )
    training_inputs = types.IngesTablesTrainingInputs(
        y_vals=torch.zeros(2, 1, 1, dtype=torch.long),
        loss_weights=torch.tensor([[[1.0]], [[0.0]]]),
    )
    classifier = head.IngesTablesClassification(
        IdentityAligner(), IdentityKvCombiner(), max_num_classes=3
    )
    optimizer = torch.optim.SGD([embeddings], lr=0.1)
    classifier.loss(classifier(embeddings, inputs), training_inputs).backward()
    torch.testing.assert_close(
        embeddings.grad[1], torch.zeros_like(embeddings[1])
    )
    optimizer.step()
    torch.testing.assert_close(embeddings[1], before[1], rtol=0, atol=0)
    self.assertFalse(torch.equal(embeddings[0], before[0]))


if __name__ == "__main__":
  absltest.main()
