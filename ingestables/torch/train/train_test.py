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

"""Tests for train.py."""

from absl.testing import absltest
from ingestables.torch.data import dummy_dataset
from ingestables.torch.model import aligner
from ingestables.torch.model import head
from ingestables.torch.model import ingestables
from ingestables.torch.model import kv_combiner
from ingestables.torch.model.backbones import vanilla_transformer
from ingestables.torch.train import train as train_lib


class TrainTest(absltest.TestCase):

  def test_train_classification(self):
    tmp_dir = self.create_tempdir()
    aligner_cls_dict = {
        "cat": aligner.CatAligner,
        "num": aligner.NumAligner,
    }
    kv_combiner_cls_dict = {
        "concat": kv_combiner.Concatenate,
    }
    backbone_cls_dict = {
        "transformer": vanilla_transformer.Transformer,
    }
    head_cls_dict = {
        "cat": head.Classification,
    }
    model_config = ingestables.ModelConfig(
        aligners={
            "cat": aligner.CatAlignerConfig(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=4,
                z_val_dim=8,
            ),
            "num": aligner.NumAlignerConfig(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=4,
                z_val_dim=8,
            ),
        },
        z_val_dim=8,
        kv_combiner_type="concat",
        backbone_type="transformer",
        backbone=vanilla_transformer.TransformerConfig(
            depth=2,
            z_dim=12,
            num_heads=1,
            dropout_attn=0.0,
            dropout_mlp=0.0,
        ),
        heads={
            "cat": head.ClassificationConfig(
                max_num_classes=2,
            ),
        },
    )
    model = ingestables.model_from_config(
        aligner_cls_dict=aligner_cls_dict,
        kv_combiner_cls_dict=kv_combiner_cls_dict,
        backbone_cls_dict=backbone_cls_dict,
        head_cls_dict=head_cls_dict,
        config=model_config,
    )
    train_lib.train_and_test(
        model=model,
        train_datasets={"xor": dummy_dataset.XorDataset()},
        test_datasets={"xor": dummy_dataset.XorDataset()},
        train_steps=100,
        train_batch_size=64,
        eval_batch_size=4,
        learning_rate=1e-2,
        momentum=0.0,
        log_loss_every_steps=20,
        workdir=tmp_dir.full_path,
        local_rank=0,
        nproc_per_node=1,
        node_rank=0,
        world_size=1,
        num_data_workers=0,
        seed=1,
    )

  def test_train_regression(self):
    tmp_dir = self.create_tempdir()
    aligner_cls_dict = {
        "cat": aligner.CatAligner,
        "num": aligner.NumAligner,
    }
    kv_combiner_cls_dict = {
        "concat": kv_combiner.Concatenate,
    }
    backbone_cls_dict = {
        "transformer": vanilla_transformer.Transformer,
    }
    head_cls_dict = {
        "num": head.Regression,
    }
    model_config = ingestables.ModelConfig(
        aligners={
            "cat": aligner.CatAlignerConfig(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=4,
                z_val_dim=8,
            ),
            "num": aligner.NumAlignerConfig(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=4,
                z_val_dim=8,
            ),
        },
        z_val_dim=8,
        kv_combiner_type="concat",
        backbone_type="transformer",
        backbone=vanilla_transformer.TransformerConfig(
            depth=2,
            z_dim=12,
            num_heads=1,
            dropout_attn=0.0,
            dropout_mlp=0.0,
        ),
        heads={
            "num": head.RegressionConfig(
                z_dim=12,
            ),
        },
    )
    model = ingestables.model_from_config(
        aligner_cls_dict=aligner_cls_dict,
        kv_combiner_cls_dict=kv_combiner_cls_dict,
        backbone_cls_dict=backbone_cls_dict,
        head_cls_dict=head_cls_dict,
        config=model_config,
    )
    train_lib.train_and_test(
        model=model,
        train_datasets={"and": dummy_dataset.AndDataset()},
        test_datasets={"and": dummy_dataset.AndDataset()},
        train_steps=100,
        train_batch_size=64,
        eval_batch_size=4,
        learning_rate=1e-2,
        momentum=0.0,
        log_loss_every_steps=20,
        workdir=tmp_dir.full_path,
        local_rank=0,
        nproc_per_node=1,
        node_rank=0,
        world_size=1,
        num_data_workers=0,
        seed=1,
    )

  def test_train_both(self):
    tmp_dir = self.create_tempdir()
    aligner_cls_dict = {
        "cat": aligner.CatAligner,
        "num": aligner.NumAligner,
    }
    kv_combiner_cls_dict = {
        "concat": kv_combiner.Concatenate,
    }
    backbone_cls_dict = {
        "transformer": vanilla_transformer.Transformer,
    }
    head_cls_dict = {
        "cat": head.Classification,
        "num": head.Regression,
    }
    model_config = ingestables.ModelConfig(
        aligners={
            "cat": aligner.CatAlignerConfig(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=8,
                z_val_dim=16,
            ),
            "num": aligner.NumAlignerConfig(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=8,
                z_val_dim=16,
            ),
        },
        z_val_dim=16,
        kv_combiner_type="concat",
        backbone_type="transformer",
        backbone=vanilla_transformer.TransformerConfig(
            depth=2,
            z_dim=24,
            num_heads=1,
            dropout_attn=0.0,
            dropout_mlp=0.0,
        ),
        heads={
            "cat": head.ClassificationConfig(
                max_num_classes=2,
            ),
            "num": head.RegressionConfig(
                z_dim=24,
            ),
        },
    )
    model = ingestables.model_from_config(
        aligner_cls_dict=aligner_cls_dict,
        kv_combiner_cls_dict=kv_combiner_cls_dict,
        backbone_cls_dict=backbone_cls_dict,
        head_cls_dict=head_cls_dict,
        config=model_config,
    )
    train_lib.train_and_test(
        model=model,
        train_datasets={
            "xor": dummy_dataset.XorDataset(),
            "and": dummy_dataset.AndDataset(),
        },
        test_datasets={
            "xor": dummy_dataset.XorDataset(),
            "and": dummy_dataset.AndDataset(),
        },
        train_steps=200,
        train_batch_size=64,
        eval_batch_size=4,
        learning_rate=1e-2,
        momentum=0.0,
        log_loss_every_steps=20,
        workdir=tmp_dir.full_path,
        local_rank=0,
        nproc_per_node=1,
        node_rank=0,
        world_size=1,
        num_data_workers=0,
        seed=1,
    )


if __name__ == "__main__":
  absltest.main()
