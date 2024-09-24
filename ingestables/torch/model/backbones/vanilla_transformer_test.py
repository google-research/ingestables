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

"""Tests for backbone.py."""

from absl.testing import absltest
from ingestables.torch.model.backbones import vanilla_transformer
import torch


class VanillaTransformerTest(absltest.TestCase):

  def test_backbone(self):
    cfg = vanilla_transformer.TransformerConfig(
        depth=1,
        z_dim=32,
        num_heads=2,
        dropout_attn=0.1,
        dropout_mlp=0.1,
    )
    transformer = vanilla_transformer.Transformer(cfg)

    batch_size = 64
    num_feats = 32
    z_dim = cfg.z_dim
    z_embs = torch.zeros(batch_size, num_feats, z_dim, dtype=torch.float32)

    z_embs_out = transformer(z_embs)

    self.assertEqual(z_embs_out.shape, (batch_size, num_feats, z_dim))


if __name__ == "__main__":
  absltest.main()
