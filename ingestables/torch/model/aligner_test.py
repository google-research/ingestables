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
from ingestables.torch.model import aligner
import torch


class AlignerTest(absltest.TestCase):

  def test_cat_aligner(self):
    x_key_dim = 768
    x_val_dim = 768
    z_key_dim = 16
    z_val_dim = 32
    cat_aligner = aligner.TextualAligner(
        x_key_dim=x_key_dim,
        x_val_dim=x_val_dim,
        z_key_dim=z_key_dim,
        z_val_dim=z_val_dim,
        key_aligner="simple",
        key_bias=False,
        key_activation_fn=None,
        val_aligner="simple",
        val_bias=False,
        val_activation_fn=None,
    )

    batch_size = 2
    num_cat_feats = 5
    x_key_emb = torch.zeros(
        batch_size, num_cat_feats, x_key_dim, dtype=torch.float32
    )
    x_val_emb = torch.zeros(
        batch_size, num_cat_feats, x_val_dim, dtype=torch.float32
    )

    z_key_embs, z_val_embs = cat_aligner(
        x_keys=x_key_emb,
        x_vals=x_val_emb,
    )

    self.assertEqual(
        z_key_embs.shape, (batch_size, num_cat_feats, z_key_dim)
    )
    self.assertEqual(
        z_val_embs.shape, (batch_size, num_cat_feats, z_val_dim)
    )

  def test_num_aligner(self):
    x_key_dim = 768
    x_val_dim = 48
    z_key_dim = 16
    z_val_dim = 32
    num_aligner = aligner.NumericAligner(
        x_key_dim=x_key_dim,
        x_val_dim=x_val_dim,
        z_key_dim=z_key_dim,
        z_val_dim=z_val_dim,
        key_aligner="simple",
        key_bias=False,
        key_activation_fn=None,
        val_aligner="simple",
        val_bias=True,
        val_activation_fn="relu",
    )

    batch_size = 2
    num_num_feats = 5
    x_key_emb = torch.zeros(
        batch_size, num_num_feats, x_key_dim, dtype=torch.float32
    )
    x_val_emb = torch.zeros(
        batch_size, num_num_feats, x_val_dim, dtype=torch.float32
    )

    z_key_embs, z_val_embs = num_aligner(
        x_keys=x_key_emb,
        x_vals=x_val_emb,
    )

    self.assertEqual(z_key_embs.shape, (batch_size, num_num_feats, z_key_dim))
    self.assertEqual(z_val_embs.shape, (batch_size, num_num_feats, z_val_dim))

  def test_periodic_num_aligner(self):
    x_key_dim = 768
    x_val_dim = 1
    z_key_dim = 16
    z_val_dim = 32
    num_aligner = aligner.NumericAligner(
        x_key_dim=x_key_dim,
        x_val_dim=x_val_dim,
        z_key_dim=z_key_dim,
        z_val_dim=z_val_dim,
        key_aligner="simple",
        key_bias=False,
        key_activation_fn=None,
        val_aligner="periodic",
        val_bias=True,
        val_activation_fn="relu",
    )

    batch_size = 2
    num_num_feats = 5
    x_key_emb = torch.zeros(
        batch_size, num_num_feats, x_key_dim, dtype=torch.float32
    )
    x_val_emb = torch.zeros(
        batch_size, num_num_feats, x_val_dim, dtype=torch.float32
    )

    z_key_embs, z_val_embs = num_aligner(
        x_keys=x_key_emb,
        x_vals=x_val_emb,
    )

    self.assertEqual(
        z_key_embs.shape, (batch_size, num_num_feats, z_key_dim)
    )
    self.assertEqual(
        z_val_embs.shape, (batch_size, num_num_feats, z_val_dim)
    )


if __name__ == "__main__":
  absltest.main()
