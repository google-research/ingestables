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
from ingestables.torch.model import ingestables
from ingestables.torch.model.lib import generic_embeddings
import torch
from torch import nn


class FakeAligner(nn.Module):

  def forward(
      self,
      x_keys: torch.Tensor,
      x_vals: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return x_keys, x_vals


class FakeKvCombiner(nn.Module):

  def forward(
      self,
      z_key_emb: torch.Tensor,
      z_val_emb: torch.Tensor,
  ) -> torch.Tensor:
    return torch.cat((z_key_emb, z_val_emb), dim=-1)


class FakeBackbone(nn.Module):

  def forward(self, z_emb: torch.Tensor) -> torch.Tensor:
    return z_emb


class FakeHead(nn.Module):

  def forward(
      self,
      z_emb: torch.Tensor,
      inference_inputs: types.IngesTablesInferenceInputs,
  ) -> torch.Tensor:
    del inference_inputs
    return torch.mean(z_emb, dim=-1, keepdim=True)

  def loss(
      self,
      logits: torch.Tensor,
      training_inputs: types.IngesTablesTrainingInputs,
  ) -> torch.Tensor:
    del training_inputs
    return torch.mean(logits)


class EncoderTest(absltest.TestCase):

  def test_mask_and_missing(self):
    z_val_emb = torch.ones(4, dtype=torch.float32)
    mask = torch.as_tensor([0, 0, 1, 1], dtype=torch.bool)
    mask_emb = torch.full((4,), fill_value=-1, dtype=torch.float32)
    missing = torch.as_tensor([0, 1, 0, 1], dtype=torch.bool)
    missing_emb = torch.full((4,), fill_value=-2, dtype=torch.float32)

    # expected[0] is 1, because both mask and missing are 1, so z_val_emb
    # passes straight through.
    # expected[1] is -2, because missing is 0, so we replace it with
    # missing_emb's value.
    # expected[2] is -1, because mask is 0, so we replace it with
    # mask_emb's value.
    # expected[3] is -1, because mask overrides missing. This is because
    # missing can be considered as "actual information", whereas mask is applied
    # during training in order to ensure that information is removed, so mask
    # takes precedence.
    expected = torch.as_tensor([1, -2, -1, -1], dtype=torch.float32)
    actual = generic_embeddings.apply_mask_and_missing(
        z_val_emb,
        mask=mask,
        mask_emb=mask_emb,
        missing=missing,
        missing_emb=missing_emb,
    )

    self.assertEqual(expected.tolist(), actual.tolist())

  def test_encoder(self):
    batch_size = 2
    num_type_1_feats = 1
    num_type_2_feats = 2
    num_type_3_feats = 3
    x_key_dim = 8
    x_val_dim = 16

    encoder = ingestables.Encoder(
        aligners={
            "feat_type_1": FakeAligner(),
            "feat_type_2": FakeAligner(),
            "feat_type_3": FakeAligner(),
        },
        special_tokens={
            "feat_type_1": generic_embeddings.IngesTablesSpecialTokens(
                x_val_dim
            ),
            "feat_type_2": generic_embeddings.IngesTablesSpecialTokens(
                x_val_dim
            ),
            "feat_type_3": generic_embeddings.IngesTablesSpecialTokens(
                x_val_dim
            ),
        },
        kv_combiner={
            "feat_type_1": FakeKvCombiner(),
            "feat_type_2": FakeKvCombiner(),
            "feat_type_3": FakeKvCombiner(),
        },
        backbone=FakeBackbone(),
    )
    inference_inputs = {
        "feat_type_1": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_type_1_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_type_1_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
        ),
        "feat_type_2": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_type_2_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_type_2_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
        ),
        "feat_type_3": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_type_3_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_type_3_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
        ),
    }
    z_emb_dict = encoder(inference_inputs)
    self.assertEqual(
        z_emb_dict.keys(), {"feat_type_1", "feat_type_2", "feat_type_3"}
    )
    self.assertEqual(
        z_emb_dict["feat_type_1"].shape,
        (batch_size, num_type_1_feats, x_key_dim + x_val_dim),
    )
    self.assertEqual(
        z_emb_dict["feat_type_2"].shape,
        (batch_size, num_type_2_feats, x_key_dim + x_val_dim),
    )
    self.assertEqual(
        z_emb_dict["feat_type_3"].shape,
        (batch_size, num_type_3_feats, x_key_dim + x_val_dim),
    )


class ModelTest(absltest.TestCase):

  def test_model(self):
    batch_size = 2
    num_type_1_feats = 1
    num_type_2_feats = 2
    num_type_3_feats = 3
    x_key_dim = 8
    x_val_dim = 16

    model = ingestables.Model(
        aligners={
            "feat_type_1": FakeAligner(),
            "feat_type_2": FakeAligner(),
            "feat_type_3": FakeAligner(),
        },
        special_tokens={
            "feat_type_1": generic_embeddings.IngesTablesSpecialTokens(
                x_val_dim
            ),
            "feat_type_2": generic_embeddings.IngesTablesSpecialTokens(
                x_val_dim
            ),
            "feat_type_3": generic_embeddings.IngesTablesSpecialTokens(
                x_val_dim
            ),
        },
        kv_combiner={
            "feat_type_1": FakeKvCombiner(),
            "feat_type_2": FakeKvCombiner(),
            "feat_type_3": FakeKvCombiner(),
        },
        backbone=FakeBackbone(),
        heads={
            "feat_type_1": FakeHead(),
            "feat_type_2": FakeHead(),
        },
    )
    inference_inputs = {
        "feat_type_1": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_type_1_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_type_1_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
        ),
        "feat_type_2": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_type_2_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_type_2_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
        ),
        "feat_type_3": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_type_3_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_type_3_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
        ),
    }
    logits_dict = model(inference_inputs)
    # Note that "feat_type_3" is absent, because there is no corresponding head.
    self.assertEqual(logits_dict.keys(), {"feat_type_1", "feat_type_2"})
    self.assertEqual(
        logits_dict["feat_type_1"].shape, (batch_size, num_type_1_feats, 1)
    )
    self.assertEqual(
        logits_dict["feat_type_2"].shape, (batch_size, num_type_2_feats, 1)
    )

    training_inputs = {
        "feat_type_2": types.IngesTablesTrainingInputs(),
        "feat_type_3": types.IngesTablesTrainingInputs(),
    }
    losses_dict = model.loss(logits_dict, training_inputs)
    # Note that "feat_type_3" is absent, because there is no corresponding
    # logits.
    # Note that "feat_type_1" is absent, because there is no corresponding
    # training input.
    self.assertEqual(losses_dict.keys(), {"feat_type_2"})
    self.assertEqual(losses_dict["feat_type_2"].shape, ())


if __name__ == "__main__":
  absltest.main()
