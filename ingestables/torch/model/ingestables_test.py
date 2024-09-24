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

"""Tests for model.py."""

from absl.testing import absltest
from ingestables.torch.model import aligner
from ingestables.torch.model import head
from ingestables.torch.model import ingestables
from ingestables.torch.model import kv_combiner
from ingestables.torch.model.backbones import vanilla_transformer as backbone_lib
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
      inference_inputs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    del inference_inputs
    return torch.mean(z_emb, dim=-1, keepdim=True)

  def loss(
      self,
      logits: torch.Tensor,
      training_inputs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    del training_inputs
    return torch.mean(logits)


class EncoderTest(absltest.TestCase):

  def test_mask_and_missing(self):
    z_val_emb = torch.ones(4, dtype=torch.float32)
    mask = torch.as_tensor([1, 1, 0, 0], dtype=torch.bool)
    mask_emb = torch.full((4,), fill_value=-1, dtype=torch.float32)
    missing = torch.as_tensor([1, 0, 1, 0], dtype=torch.bool)
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
    actual = ingestables.apply_mask_and_missing(
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
        kv_combiner=FakeKvCombiner(),
        backbone=FakeBackbone(),
        mask_emb=nn.Parameter(torch.zeros(x_val_dim, dtype=torch.float32)),
        missing_emb=nn.Parameter(torch.zeros(x_val_dim, dtype=torch.float32)),
    )
    inference_inputs = {
        "feat_type_1": {
            "x_keys": torch.zeros(
                batch_size,
                num_type_1_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_type_1_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
        },
        "feat_type_2": {
            "x_keys": torch.zeros(
                batch_size,
                num_type_2_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_type_2_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
        },
        "feat_type_3": {
            "x_keys": torch.zeros(
                batch_size,
                num_type_3_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_type_3_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
        },
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

  def test_encoder_from_config(self):

    text_emb_dim = 768
    num_emb_dim = 48
    z_key_dim = 16
    z_val_dim = 32
    num_heads = 2
    depth = 1
    dropout_attn = 0.0
    dropout_mlp = 0.0

    aligner_cls_dict = {
        "cat": aligner.CatAligner,
        "num": aligner.NumAligner,
    }
    kv_combiner_cls_dict = {
        "concat": kv_combiner.Concatenate,
    }
    backbone_cls_dict = {
        "transformer": backbone_lib.Transformer,
    }
    config = ingestables.EncoderConfig(
        aligners={
            "cat": aligner.CatAlignerConfig(
                x_key_dim=text_emb_dim,
                x_val_dim=text_emb_dim,
                z_key_dim=z_key_dim,
                z_val_dim=z_val_dim,
            ),
            "num": aligner.NumAlignerConfig(
                x_key_dim=text_emb_dim,
                x_val_dim=num_emb_dim,
                z_key_dim=z_key_dim,
                z_val_dim=z_val_dim,
            ),
        },
        z_val_dim=z_val_dim,
        kv_combiner_type="concat",
        backbone_type="transformer",
        backbone=backbone_lib.TransformerConfig(
            depth=depth,
            z_dim=z_key_dim + z_val_dim,
            num_heads=num_heads,
            dropout_attn=dropout_attn,
            dropout_mlp=dropout_mlp,
        ),
    )
    ingestables_encoder = ingestables.encoder_from_config(
        aligner_cls_dict=aligner_cls_dict,
        kv_combiner_cls_dict=kv_combiner_cls_dict,
        backbone_cls_dict=backbone_cls_dict,
        config=config,
    )
    self.assertIsInstance(ingestables_encoder, ingestables.Encoder)

    batch_size = 2
    num_cat_feats = 3
    num_num_feats = 4

    inference_inputs = {
        "cat": {
            "x_keys": torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
        },
        "num": {
            "x_keys": torch.zeros(
                batch_size,
                num_num_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_num_feats,
                num_emb_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
        },
    }
    z_emb_dict = ingestables_encoder(inference_inputs)
    self.assertEqual(z_emb_dict.keys(), {"cat", "num"})
    self.assertEqual(
        z_emb_dict["cat"].shape,
        (batch_size, num_cat_feats, z_key_dim + z_val_dim),
    )
    self.assertEqual(
        z_emb_dict["num"].shape,
        (batch_size, num_num_feats, z_key_dim + z_val_dim),
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
        kv_combiner=FakeKvCombiner(),
        backbone=FakeBackbone(),
        heads={
            "feat_type_1": FakeHead(),
            "feat_type_2": FakeHead(),
        },
        mask_emb=nn.Parameter(torch.zeros(x_val_dim, dtype=torch.float32)),
        missing_emb=nn.Parameter(torch.zeros(x_val_dim, dtype=torch.float32)),
    )
    inference_inputs = {
        "feat_type_1": {
            "x_keys": torch.zeros(
                batch_size,
                num_type_1_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_type_1_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_type_1_feats,
                1,
                dtype=torch.bool,
            ),
        },
        "feat_type_2": {
            "x_keys": torch.zeros(
                batch_size,
                num_type_2_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_type_2_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_type_2_feats,
                1,
                dtype=torch.bool,
            ),
        },
        "feat_type_3": {
            "x_keys": torch.zeros(
                batch_size,
                num_type_3_feats,
                x_key_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_type_3_feats,
                x_val_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_type_3_feats,
                1,
                dtype=torch.bool,
            ),
        },
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
        "feat_type_2": {},
        "feat_type_3": {},
    }
    losses_dict = model.loss(logits_dict, training_inputs)
    # Note that "feat_type_3" is absent, because there is no corresponding
    # logits.
    # Note that "feat_type_1" is absent, because there is no corresponding
    # training input.
    self.assertEqual(losses_dict.keys(), {"feat_type_2"})
    self.assertEqual(losses_dict["feat_type_2"].shape, ())

  def test_model_from_config(self):
    batch_size = 4
    text_emb_dim = 768
    num_emb_dim = 48
    max_num_classes = 2
    z_key_dim = 16
    z_val_dim = 32
    num_heads = 2
    depth = 1
    dropout_attn = 0.0
    dropout_mlp = 0.0

    aligner_cls_dict = {
        "cat": aligner.CatAligner,
        "num": aligner.NumAligner,
    }
    kv_combiner_cls_dict = {
        "concat": kv_combiner.Concatenate,
    }
    backbone_cls_dict = {
        "transformer": backbone_lib.Transformer,
    }
    head_cls_dict = {
        "cat": head.Classification,
        "num": head.Regression,
    }
    config = ingestables.ModelConfig(
        aligners={
            "cat": aligner.CatAlignerConfig(
                x_key_dim=text_emb_dim,
                x_val_dim=text_emb_dim,
                z_key_dim=z_key_dim,
                z_val_dim=z_val_dim,
            ),
            "num": aligner.NumAlignerConfig(
                x_key_dim=text_emb_dim,
                x_val_dim=num_emb_dim,
                z_key_dim=z_key_dim,
                z_val_dim=z_val_dim,
            ),
        },
        z_val_dim=z_val_dim,
        kv_combiner_type="concat",
        backbone_type="transformer",
        backbone=backbone_lib.TransformerConfig(
            depth=depth,
            z_dim=z_key_dim + z_val_dim,
            num_heads=num_heads,
            dropout_attn=dropout_attn,
            dropout_mlp=dropout_mlp,
        ),
        heads={
            "cat": head.ClassificationConfig(
                max_num_classes=max_num_classes,
            ),
            "num": head.RegressionConfig(z_dim=z_key_dim + z_val_dim),
        },
    )
    model = ingestables.model_from_config(
        aligner_cls_dict=aligner_cls_dict,
        kv_combiner_cls_dict=kv_combiner_cls_dict,
        backbone_cls_dict=backbone_cls_dict,
        head_cls_dict=head_cls_dict,
        config=config,
    )
    self.assertIsInstance(model, ingestables.Model)

    num_cat_feats = 5
    num_num_feats = 7
    inference_inputs = {
        "cat": {
            "x_keys": torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
            "x_vals_all": torch.zeros(
                batch_size,
                num_cat_feats,
                max_num_classes,
                text_emb_dim,
                dtype=torch.float32,
            ),
            "padding": torch.zeros(
                batch_size,
                num_cat_feats,
                max_num_classes,
                dtype=torch.bool,
            ),
        },
        "num": {
            "x_keys": torch.zeros(
                batch_size,
                num_num_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            "x_vals": torch.zeros(
                batch_size,
                num_num_feats,
                num_emb_dim,
                dtype=torch.float32,
            ),
            "mask": torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
            "missing": torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
        },
    }
    logits_dict = model(inference_inputs)
    self.assertEqual(logits_dict.keys(), {"cat", "num"})
    self.assertEqual(
        logits_dict["cat"].shape, (batch_size, num_cat_feats, max_num_classes)
    )
    self.assertEqual(logits_dict["num"].shape, (batch_size, num_num_feats, 1))

    training_inputs = {
        "num": {
            "y_vals": torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.float32,
            ),
            "loss_weights": torch.ones(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.float32,
            ),
        },
    }
    losses_dict = model.loss(logits_dict, training_inputs)
    # Note that "cat" is absent, since there is no corresponding training input.
    self.assertEqual(losses_dict.keys(), {"num"})
    self.assertEqual(losses_dict["num"].shape, ())

if __name__ == "__main__":
  absltest.main()
