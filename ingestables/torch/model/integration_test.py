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

"""Integration test."""

from absl.testing import absltest
import fiddle as fdl
from ingestables.torch import types
from ingestables.torch.model import aligner
from ingestables.torch.model import head
from ingestables.torch.model import ingestables
from ingestables.torch.model import kv_combiner as kv_combiner_lib
from ingestables.torch.model.backbones import vanilla_transformer
from ingestables.torch.model.lib import generic_embeddings
import torch


def get_encoder_config(
    text_emb_dim: int,
    num_emb_dim: int,
    depth: int,
    z_key_dim: int,
    z_val_dim: int,
    num_heads: int,
    dropout_attn: float,
    dropout_mlp: float,
) -> fdl.Config[ingestables.Encoder]:
  """Makes a config for ingestables.Encoder."""
  encoder_config = fdl.Config(
      ingestables.Encoder,
      aligners={
          "cat": fdl.Config(
              aligner.TextualAligner,
              x_key_dim=text_emb_dim,
              x_val_dim=text_emb_dim,
              z_key_dim=z_key_dim,
              z_val_dim=z_val_dim,
              key_aligner="simple",
              key_bias=False,
              key_activation_fn=None,
              val_aligner="simple",
              val_bias=False,
              val_activation_fn=None,
          ),
          "num": fdl.Config(
              aligner.NumericAligner,
              x_key_dim=text_emb_dim,
              x_val_dim=num_emb_dim,
              z_key_dim=z_key_dim,
              z_val_dim=z_val_dim,
              key_aligner="simple",
              key_bias=False,
              key_activation_fn=None,
              val_aligner="simple",
              val_bias=False,
              val_activation_fn=None,
          ),
          "str": fdl.Config(
              aligner.TextualAligner,
              x_key_dim=text_emb_dim,
              x_val_dim=text_emb_dim,
              z_key_dim=z_key_dim,
              z_val_dim=z_val_dim,
              key_aligner="simple",
              key_bias=False,
              key_activation_fn=None,
              val_aligner="simple",
              val_bias=False,
              val_activation_fn=None,
          ),
      },
      special_tokens={
          "cat": fdl.Config(
              generic_embeddings.IngesTablesSpecialTokens, z_val_dim=z_val_dim
          ),
          "num": fdl.Config(
              generic_embeddings.IngesTablesSpecialTokens, z_val_dim=z_val_dim
          ),
          "str": fdl.Config(
              generic_embeddings.IngesTablesSpecialTokens, z_val_dim=z_val_dim
          ),
      },
      kv_combiner={
          "cat": fdl.Config(kv_combiner_lib.Concatenate),
          "num": fdl.Config(kv_combiner_lib.Concatenate),
          "str": fdl.Config(kv_combiner_lib.Concatenate),
      },
      backbone=fdl.Config(
          vanilla_transformer.Transformer,
          layers=[
              fdl.Config(
                  vanilla_transformer.TransformerLayer,
                  z_dim=z_key_dim + z_val_dim,
                  num_heads=num_heads,
                  dropout_attn=dropout_attn,
                  dropout_mlp=dropout_mlp,
              )
              for _ in range(depth)
          ],
      ),
  )
  return encoder_config


def get_model_config(
    text_emb_dim: int,
    num_emb_dim: int,
    depth: int,
    z_key_dim: int,
    z_val_dim: int,
    num_heads: int,
    max_num_classes: int,
    dropout_attn: float,
    dropout_mlp: float,
) -> fdl.Config[ingestables.Model]:
  """Makes a config for ingestables.Model."""
  # We reuse the same cat_aligner and kv_combiner instances in
  # head.IngesTablesClassification.
  cat_aligner = fdl.Config(
      aligner.TextualAligner,
      x_key_dim=text_emb_dim,
      x_val_dim=text_emb_dim,
      z_key_dim=z_key_dim,
      z_val_dim=z_val_dim,
      key_aligner="simple",
      key_bias=False,
      key_activation_fn=None,
      val_aligner="simple",
      val_bias=False,
      val_activation_fn=None,
  )
  kv_combiner = fdl.Config(
      kv_combiner_lib.Concatenate,
  )
  model_config = fdl.Config(
      ingestables.Model,
      aligners={
          "cat": cat_aligner,
          "num": fdl.Config(
              aligner.NumericAligner,
              x_key_dim=text_emb_dim,
              x_val_dim=num_emb_dim,
              z_key_dim=z_key_dim,
              z_val_dim=z_val_dim,
              key_aligner="simple",
              key_bias=False,
              key_activation_fn=None,
              val_aligner="simple",
              val_bias=False,
              val_activation_fn=None,
          ),
          "str": fdl.Config(
              aligner.TextualAligner,
              x_key_dim=text_emb_dim,
              x_val_dim=text_emb_dim,
              z_key_dim=z_key_dim,
              z_val_dim=z_val_dim,
              key_aligner="simple",
              key_bias=False,
              key_activation_fn=None,
              val_aligner="simple",
              val_bias=False,
              val_activation_fn=None,
          ),
      },
      special_tokens={
          "cat": fdl.Config(
              generic_embeddings.IngesTablesSpecialTokens, z_val_dim=z_val_dim
          ),
          "num": fdl.Config(
              generic_embeddings.IngesTablesSpecialTokens, z_val_dim=z_val_dim
          ),
          "str": fdl.Config(
              generic_embeddings.IngesTablesSpecialTokens, z_val_dim=z_val_dim
          ),
      },
      kv_combiner={
          "cat": fdl.Config(kv_combiner_lib.Concatenate),
          "num": fdl.Config(kv_combiner_lib.Concatenate),
          "str": fdl.Config(kv_combiner_lib.Concatenate),
      },
      backbone=fdl.Config(
          vanilla_transformer.Transformer,
          layers=[
              fdl.Config(
                  vanilla_transformer.TransformerLayer,
                  z_dim=z_key_dim + z_val_dim,
                  num_heads=num_heads,
                  dropout_attn=dropout_attn,
                  dropout_mlp=dropout_mlp,
              )
              for _ in range(depth)
          ],
      ),
      heads={
          "cat": fdl.Config(
              head.IngesTablesClassification,
              aligner=cat_aligner,
              kv_combiner=kv_combiner,
              max_num_classes=max_num_classes,
          ),
          "num": fdl.Config(
              head.IngesTablesRegression,
              z_dim=z_key_dim + z_val_dim,
          ),
      },
  )
  return model_config


class IntegrationTest(absltest.TestCase):

  def test_encoder_from_config(self):

    text_emb_dim = 768
    num_emb_dim = 48
    z_key_dim = 16
    z_val_dim = 32

    cfg = get_encoder_config(
        text_emb_dim=text_emb_dim,
        num_emb_dim=num_emb_dim,
        depth=1,
        z_key_dim=z_key_dim,
        z_val_dim=z_val_dim,
        num_heads=2,
        dropout_attn=0.0,
        dropout_mlp=0.0,
    )
    encoder = fdl.build(cfg)
    self.assertIsInstance(encoder, ingestables.Encoder)

    batch_size = 2
    num_cat_feats = 3
    num_num_feats = 4
    num_str_feats = 5

    inference_inputs = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_num_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_num_feats,
                num_emb_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
        ),
        "str": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_str_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_str_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_str_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_str_feats,
                1,
                dtype=torch.bool,
            ),
        ),
    }
    z_emb_dict = encoder(inference_inputs)
    self.assertEqual(z_emb_dict.keys(), {"cat", "num", "str"})
    self.assertEqual(
        z_emb_dict["cat"].shape,
        (batch_size, num_cat_feats, z_key_dim + z_val_dim),
    )
    self.assertEqual(
        z_emb_dict["num"].shape,
        (batch_size, num_num_feats, z_key_dim + z_val_dim),
    )
    self.assertEqual(
        z_emb_dict["str"].shape,
        (batch_size, num_str_feats, z_key_dim + z_val_dim),
    )

  def test_model_from_config(self):

    text_emb_dim = 768
    num_emb_dim = 48
    max_num_classes = 2
    cfg = get_model_config(
        text_emb_dim=text_emb_dim,
        num_emb_dim=num_emb_dim,
        depth=1,
        z_key_dim=16,
        z_val_dim=32,
        max_num_classes=max_num_classes,
        num_heads=2,
        dropout_attn=0.0,
        dropout_mlp=0.0,
    )
    model = fdl.build(cfg)

    self.assertIsInstance(model, ingestables.Model)

    batch_size = 4
    num_cat_feats = 5
    num_num_feats = 7
    num_str_feats = 6
    inference_inputs = {
        "cat": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_cat_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_cat_feats,
                1,
                dtype=torch.bool,
            ),
            x_vals_all=torch.zeros(
                batch_size,
                num_cat_feats,
                max_num_classes,
                text_emb_dim,
                dtype=torch.float32,
            ),
            padding=torch.zeros(
                batch_size,
                num_cat_feats,
                max_num_classes,
                dtype=torch.bool,
            ),
        ),
        "num": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_num_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_num_feats,
                num_emb_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.bool,
            ),
        ),
        "str": types.IngesTablesInferenceInputs(
            x_keys=torch.zeros(
                batch_size,
                num_str_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            x_vals=torch.zeros(
                batch_size,
                num_str_feats,
                text_emb_dim,
                dtype=torch.float32,
            ),
            mask=torch.zeros(
                batch_size,
                num_str_feats,
                1,
                dtype=torch.bool,
            ),
            missing=torch.zeros(
                batch_size,
                num_str_feats,
                1,
                dtype=torch.bool,
            ),
            x_vals_all=torch.zeros(
                batch_size,
                num_str_feats,
                max_num_classes,
                text_emb_dim,
                dtype=torch.float32,
            ),
            padding=torch.zeros(
                batch_size,
                num_str_feats,
                max_num_classes,
                dtype=torch.bool,
            ),
        ),
    }
    logits_dict = model(inference_inputs)
    self.assertEqual(logits_dict.keys(), {"cat", "num"})
    self.assertEqual(
        logits_dict["cat"].shape, (batch_size, num_cat_feats, max_num_classes)
    )
    self.assertEqual(logits_dict["num"].shape, (batch_size, num_num_feats, 1))

    training_inputs = {
        "num": types.IngesTablesTrainingInputs(
            y_vals=torch.zeros(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.float32,
            ),
            loss_weights=torch.ones(
                batch_size,
                num_num_feats,
                1,
                dtype=torch.float32,
            ),
        ),
    }
    losses_dict = model.loss(logits_dict, training_inputs)
    # Note that "cat" is absent, since there is no corresponding training input.
    self.assertEqual(losses_dict.keys(), {"num"})
    self.assertEqual(losses_dict["num"].shape, ())


if __name__ == "__main__":
  absltest.main()
