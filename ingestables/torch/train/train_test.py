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

import shutil
import tempfile

from absl.testing import absltest
from ingestables.torch.data import encoders
from ingestables.torch.data import pipeline as pipeline_lib
from ingestables.torch.data import preprocessors
from ingestables.torch.data import scenario_generators
from ingestables.torch.model import aligner
from ingestables.torch.model import head
from ingestables.torch.model import ingestables
from ingestables.torch.model import kv_combiner as kv_combiner_lib
from ingestables.torch.model import text_encoders
from ingestables.torch.model.backbones import vanilla_transformer
from ingestables.torch.model.lib import generic_embeddings
from ingestables.torch.model.lib import masking
from ingestables.torch.train import lr_scheduler
from ingestables.torch.train import metrics
from ingestables.torch.train import train as train_lib
import torch


ROOT_DIR = "~/ingestables/"


class TrainTest(absltest.TestCase):
  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tmp_dir)

  def test_train_classification(self):
    cat_aligner = aligner.TextualAligner(
        x_key_dim=3,
        x_val_dim=3,
        z_key_dim=8,
        z_val_dim=16,
        key_aligner="simple",
        key_bias=False,
        key_activation_fn=None,
        val_aligner="simple",
        val_bias=False,
        val_activation_fn=None,
    )
    kv_combiner = kv_combiner_lib.Concatenate()
    model = ingestables.Model(
        aligners={
            "cat": cat_aligner,
            "num": aligner.NumericAligner(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=8,
                z_val_dim=16,
                key_aligner="simple",
                key_bias=False,
                key_activation_fn=None,
                val_aligner="simple",
                val_bias=False,
                val_activation_fn=None,
            ),
        },
        special_tokens={
            "cat": generic_embeddings.IngesTablesSpecialTokens(z_val_dim=16),
            "num": generic_embeddings.IngesTablesSpecialTokens(z_val_dim=16),
        },
        kv_combiner={"cat": kv_combiner, "num": kv_combiner},
        backbone=vanilla_transformer.Transformer(
            layers=[
                vanilla_transformer.TransformerLayer(
                    z_dim=24,
                    num_heads=1,
                    dropout_attn=0.1,
                    dropout_mlp=0.1,
                )
                for _ in range(2)
            ],
        ),
        heads={
            "cat": head.IngesTablesClassification(
                aligner=cat_aligner,
                kv_combiner=kv_combiner,
                max_num_classes=2,
            ),
        },
    )
    metrics_store = metrics.InMemoryMetricsStore()
    metrics_writer = metrics.MetricsWriter(metrics_store)
    num_train_steps = 100
    peak_lr = 3e-3
    trainer = train_lib.Trainer(  # pylint: disable=unused-variable
        workdir=self.tmp_dir,
        model=model,
        optimizer=lambda params: torch.optim.Adam(params, lr=peak_lr),
        lr_scheduler=lambda optim: lr_scheduler.LinearWarmupCosineLRScheduler(
            optim,
            max_steps=num_train_steps,
            warmup_steps=num_train_steps // 10,
            warmup_start_lr=0,
            warmup_end_lr=peak_lr,
            peak_lr=peak_lr,
            min_lr=peak_lr * 0.5,
        ),
        pipeline=pipeline_lib.Pipeline(
            pipeline_modules=[
                pipeline_lib.PipelineModule(
                    benchmark_name="test",
                    dataset_name="xor",
                    splitter=scenario_generators.Splitter(random_state=13),
                    sampler=scenario_generators.Sampler(random_state=13),
                    preprocessor=preprocessors.Preprocessor(
                        noise=0.0,
                        numeric_scaling_method="min-max",
                    ),
                    encoder=encoders.Encoder(
                        max_num_categories=3,
                        n_bins=3,
                        target_encoding="llm",
                        batch_size=32,
                        text_encoder=text_encoders.TextEncoder(
                            text_encoder_name="hash",
                            # [NOTE] MUST match the embedding_dim of the
                            # backbone.
                            embedding_dim=3,
                        ),
                    ),
                ),
            ],
        ),
        num_train_steps=num_train_steps,
        log_loss_every_steps=num_train_steps // 5,
        eval_every_steps=num_train_steps // 5,
        train_batch_size=8,
        eval_batch_size=8,
        checkpoint_every_steps=num_train_steps,
        num_data_workers=0,
        amp_dtype=None,
        metrics_writer=metrics_writer,
        masking_strategy=masking.MaskingStrategy(
            target_masking_prob=1.0, default_masking_prob=0.0, random_seed=13
        ),
        load_backbone_weights=False,
        load_aligner_weights_if_available=False,
        load_head_weights_if_available=False,
        freeze_backbone=False,
        freeze_aligners=False,
        freeze_heads=False,
        freeze_special_tokens=False,
        prefetch_factor=None,
        enable_amp=False,
    )

    trainer.run()

    self.assertGreater(
        metrics_store.metric(
            step=num_train_steps, key=("xor", "cat"), metric_name="val_auc_pr"
        ),
        0.9,
    )
    self.assertGreater(
        metrics_store.metric(
            step=num_train_steps, key=("xor", "cat"), metric_name="test_auc_pr"
        ),
        0.9,
    )

  def test_train_regression(self):
    model = ingestables.Model(
        aligners={
            "cat": aligner.TextualAligner(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=8,
                z_val_dim=16,
                key_aligner="simple",
                key_bias=False,
                key_activation_fn=None,
                val_aligner="simple",
                val_bias=False,
                val_activation_fn=None,
            ),
            "num": aligner.NumericAligner(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=8,
                z_val_dim=16,
                key_aligner="simple",
                key_bias=False,
                key_activation_fn=None,
                val_aligner="simple",
                val_bias=False,
                val_activation_fn=None,
            ),
        },
        special_tokens={
            "cat": generic_embeddings.IngesTablesSpecialTokens(z_val_dim=16),
            "num": generic_embeddings.IngesTablesSpecialTokens(z_val_dim=16),
        },
        kv_combiner={
            "cat": kv_combiner_lib.Concatenate(),
            "num": kv_combiner_lib.Concatenate(),
        },
        backbone=vanilla_transformer.Transformer(
            layers=[
                vanilla_transformer.TransformerLayer(
                    z_dim=24,
                    num_heads=1,
                    dropout_attn=0.0,
                    dropout_mlp=0.0,
                )
                for _ in range(2)
            ],
        ),
        heads={
            "num": head.IngesTablesRegression(
                z_dim=24,
            ),
        },
    )

    metrics_store = metrics.InMemoryMetricsStore()
    metrics_writer = metrics.MetricsWriter(metrics_store)
    num_train_steps = 100
    peak_lr = 3e-3
    trainer = train_lib.Trainer(  # pylint: disable=unused-variable
        workdir=self.tmp_dir,
        model=model,
        optimizer=lambda params: torch.optim.Adam(params, lr=peak_lr),
        lr_scheduler=lambda optim: lr_scheduler.LinearWarmupCosineLRScheduler(
            optim,
            max_steps=num_train_steps,
            warmup_steps=num_train_steps // 10,
            warmup_start_lr=0,
            warmup_end_lr=peak_lr,
            peak_lr=peak_lr,
            min_lr=peak_lr * 0.5,
        ),
        pipeline=pipeline_lib.Pipeline(
            pipeline_modules=[
                pipeline_lib.PipelineModule(
                    benchmark_name="test",
                    dataset_name="and",
                    splitter=scenario_generators.Splitter(random_state=13),
                    sampler=scenario_generators.Sampler(random_state=13),
                    preprocessor=preprocessors.Preprocessor(
                        noise=0.0,
                        numeric_scaling_method="min-max",
                    ),
                    encoder=encoders.Encoder(
                        max_num_categories=3,
                        n_bins=3,
                        batch_size=32,
                        text_encoder=text_encoders.TextEncoder(
                            text_encoder_name="hash",
                            # [NOTE] MUST match the embedding_dim of the
                            # backbone.
                            embedding_dim=3,
                        ),
                    ),
                ),
            ],
        ),
        num_train_steps=num_train_steps,
        log_loss_every_steps=num_train_steps // 5,
        eval_every_steps=num_train_steps // 5,
        checkpoint_every_steps=num_train_steps // 5,
        num_data_workers=0,
        amp_dtype=None,
        train_batch_size=8,
        eval_batch_size=8,
        metrics_writer=metrics_writer,
        masking_strategy=masking.MaskingStrategy(
            target_masking_prob=1.0, default_masking_prob=0.0, random_seed=13
        ),
        load_backbone_weights=False,
        load_aligner_weights_if_available=False,
        load_head_weights_if_available=False,
        freeze_backbone=False,
        freeze_aligners=False,
        freeze_heads=False,
        freeze_special_tokens=False,
        prefetch_factor=None,
        enable_amp=False,
    )

    trainer.run()

    self.assertLess(
        metrics_store.metric(
            num_train_steps, ("and", "num"), "val_mean_squared_error"
        ),
        0.5,
    )
    self.assertLess(
        metrics_store.metric(
            num_train_steps, ("and", "num"), "test_mean_squared_error"
        ),
        0.5,
    )

  def test_train_both(self):
    cat_aligner = aligner.TextualAligner(
        x_key_dim=3,
        x_val_dim=3,
        z_key_dim=8,
        z_val_dim=16,
        key_aligner="simple",
        key_bias=False,
        key_activation_fn=None,
        val_aligner="simple",
        val_bias=False,
        val_activation_fn=None,
    )
    kv_combiner = kv_combiner_lib.Concatenate()
    model = ingestables.Model(
        aligners={
            "cat": cat_aligner,
            "num": aligner.NumericAligner(
                x_key_dim=3,
                x_val_dim=3,
                z_key_dim=8,
                z_val_dim=16,
                key_aligner="simple",
                key_bias=False,
                key_activation_fn=None,
                val_aligner="simple",
                val_bias=False,
                val_activation_fn=None,
            ),
        },
        special_tokens={
            "cat": generic_embeddings.IngesTablesSpecialTokens(z_val_dim=16),
            "num": generic_embeddings.IngesTablesSpecialTokens(z_val_dim=16),
        },
        kv_combiner={
            "cat": kv_combiner_lib.Concatenate(),
            "num": kv_combiner_lib.Concatenate(),
        },
        backbone=vanilla_transformer.Transformer(
            layers=[
                vanilla_transformer.TransformerLayer(
                    z_dim=24,
                    num_heads=1,
                    dropout_attn=0.1,
                    dropout_mlp=0.1,
                )
                for _ in range(2)
            ],
        ),
        heads={
            "cat": head.IngesTablesClassification(
                aligner=cat_aligner,
                kv_combiner=kv_combiner,
                max_num_classes=2,
            ),
            "num": head.IngesTablesRegression(z_dim=24),
        },
    )
    metrics_store = metrics.InMemoryMetricsStore()
    metrics_writer = metrics.MetricsWriter(metrics_store)
    num_train_steps = 200
    peak_lr = 3e-3
    trainer = train_lib.Trainer(  # pylint: disable=unused-variable
        workdir=self.tmp_dir,
        model=model,
        optimizer=lambda params: torch.optim.Adam(params, lr=peak_lr),
        lr_scheduler=lambda optim: lr_scheduler.LinearWarmupCosineLRScheduler(
            optim,
            max_steps=num_train_steps,
            warmup_steps=num_train_steps // 10,
            warmup_start_lr=0,
            warmup_end_lr=peak_lr,
            peak_lr=peak_lr,
            min_lr=peak_lr * 0.5,
        ),
        pipeline=pipeline_lib.Pipeline(
            pipeline_modules=[
                pipeline_lib.PipelineModule(
                    benchmark_name="test",
                    dataset_name="xor",
                    splitter=scenario_generators.Splitter(random_state=13),
                    sampler=scenario_generators.Sampler(random_state=13),
                    preprocessor=preprocessors.Preprocessor(
                        noise=0.0,
                        numeric_scaling_method="min-max",
                    ),
                    encoder=encoders.Encoder(
                        max_num_categories=3,
                        target_encoding="llm",
                        n_bins=3,
                        batch_size=32,
                        text_encoder=text_encoders.TextEncoder(
                            text_encoder_name="hash",
                            # [NOTE] MUST match the embedding_dim of the
                            # backbone.
                            embedding_dim=3,
                        ),
                    ),
                ),
                pipeline_lib.PipelineModule(
                    benchmark_name="test",
                    dataset_name="and",
                    splitter=scenario_generators.Splitter(random_state=13),
                    sampler=scenario_generators.Sampler(random_state=13),
                    preprocessor=preprocessors.Preprocessor(
                        noise=0.0,
                        numeric_scaling_method="min-max",
                    ),
                    encoder=encoders.Encoder(
                        max_num_categories=3,
                        n_bins=3,
                        batch_size=32,
                        text_encoder=text_encoders.TextEncoder(
                            text_encoder_name="hash",
                            # [NOTE] MUST match the embedding_dim of the
                            # backbone.
                            embedding_dim=3,
                        ),
                    ),
                ),
            ],
        ),
        num_train_steps=num_train_steps,
        train_batch_size=8,
        eval_batch_size=8,  # TODO(joetoth): Bug when eval =! train.
        log_loss_every_steps=num_train_steps // 5,
        eval_every_steps=num_train_steps // 5,
        checkpoint_every_steps=num_train_steps,
        num_data_workers=0,
        amp_dtype=None,
        metrics_writer=metrics_writer,
        masking_strategy=masking.MaskingStrategy(
            target_masking_prob=1.0, default_masking_prob=0.0, random_seed=13
        ),
        load_backbone_weights=False,
        load_aligner_weights_if_available=False,
        load_head_weights_if_available=False,
        freeze_backbone=False,
        freeze_aligners=False,
        freeze_heads=False,
        freeze_special_tokens=False,
        prefetch_factor=None,
        enable_amp=False,
    )

    trainer.run()

    self.assertGreater(
        metrics_store.metric(
            step=num_train_steps, key=("xor", "cat"), metric_name="val_auc_pr"
        ),
        0.9,
    )
    self.assertGreater(
        metrics_store.metric(
            step=num_train_steps, key=("xor", "cat"), metric_name="test_auc_pr"
        ),
        0.9,
    )
    self.assertLess(
        metrics_store.metric(
            num_train_steps, ("and", "num"), "val_mean_squared_error"
        ),
        0.5,
    )
    self.assertLess(
        metrics_store.metric(
            num_train_steps, ("and", "num"), "test_mean_squared_error"
        ),
        0.5,
    )


if __name__ == "__main__":
  absltest.main()
