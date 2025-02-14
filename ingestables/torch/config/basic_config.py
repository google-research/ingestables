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

"""Fiddle config for running the simplest experiment."""

import copy
import functools
import logging
from typing import List, Literal, Tuple

from etils import epath
from fiddle.experimental import auto_config
from ingestables.torch import tabular_foundation_lab
from ingestables.torch.data import encoders
from ingestables.torch.data import pipeline
from ingestables.torch.data import preprocessors
from ingestables.torch.data import scenario_generators
from ingestables.torch.data import serialization
from ingestables.torch.google.baselines import ft_transformer
from ingestables.torch.google.baselines import mlp
from ingestables.torch.google.baselines import resnet
from ingestables.torch.model import aligner
from ingestables.torch.model import head
from ingestables.torch.model import ingestables
from ingestables.torch.model import kv_combiner as kv_combiner_lib
from ingestables.torch.model import sklearn_model
from ingestables.torch.model import text_encoders
from ingestables.torch.model.backbones import t5_transformer
from ingestables.torch.model.lib import generic_embeddings
from ingestables.torch.model.lib import masking
from ingestables.torch.train import lr_scheduler
from ingestables.torch.train import metrics
from ingestables.torch.train import train
from ingestables.torch.train import train_deep
from ingestables.torch.train import train_sklearn
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
import torch
import tqdm
import transformers
import xgboost

ROOT_DIR = "~/ingestables/"

# The paths to pre-trained IngesTables models go here.
# Naming Convention:
# <Pre-training corpus name>-<Backbone Size>-<Text Encoder Size>
PRETRAINED_INGESTABLES_PATHS = {}


GENERIC_PRETRAINING_DATASET_NAMES = (
    "anime_planet",
    "babies_r_us",
    "bikedekho",
    "bikewale",
    "buy_buy_baby",
    "cardekho",
    "chocolate_bar_ratings",
    "clear_corpus",
    "employee_remuneration",  # Repeated
    "employee_salaries",  # Repeated
    "filmtv_movies",
    "jp_anime",
    "movies",
    "nba_draft",
    "prescription_drugs",
    "rotten_tomatoes",
    "spotify",
    "us_accidents_counts",  # Repeated
    "us_accidents_severity",  # Repeated
    "used_cars_pakistan",
    "videogame_sales",
)

ALCOHOL_PRETRAINING_DATASET_NAMES = (
    "wikiliq_beer",
    "wikiliq_spirit",
    "wina_pl",
    "wine_dot_com_prices",
    "wine_dot_com_ratings",
    "wine_enthusiasts_prices",
    "wine_enthusiasts_ratings",
)

HOUSING_PRETRAINING_DATASET_NAMES = (
    "us_real_estate",
    "usa_housing",
    "us_airbnb",
    "nashville_housing",
)

CREDIT_PRETRAINING_DATASET_NAMES = (
    "autos",
    "give_me_some_credit",
    "south_africa_debt",
    "indonesian_telecom_delinquency",
)

RESTAURANT_PRETRAINING_DATASET_NAMES = ("michelin", "yelp")

text_embedding_size_to_dim = {
    "base": 768,
    "large": 1024,
    "xl": 1024,
    "xxl": 1024,
}

backbone_size_to_dim = {
    "tiny": 256,
    "mini": 384,
    "small": 512,
    "base": 512,
    "large": 1024,
}


@auto_config.auto_config
def full_base_config(
    benchmark_name: str = "carte",
    dataset_name: str = "cardekho",
    num_train_steps: int = 5000,
    text_embedding_size: Literal["base", "large", "xl", "xxl"] = "xl",
    backbone_size: Literal["tiny", "mini", "small", "base", "large"] = "tiny",
    z_key_to_val_dim_ratio: str = "1:3",
    target_masking_prob: float = 1.0,
    default_masking_prob: float = 0.3,
    ingestables_checkpoint_name_and_path: str | None = None,
    load_aligner_weights_if_available: bool = False,
    load_backbone_weights: bool = True,
    load_head_weights_if_available: bool = False,
    freeze_backbone: bool = False,
    freeze_aligners: bool = False,
    freeze_heads: bool = False,
    freeze_special_tokens: bool = False,
    learning_rate: float = 3e-4,
    warmup_steps: int = 100,
) -> tabular_foundation_lab.TablularFoundationLab:
  """Makes a basic fiddle config for ingestables.Model."""

  text_embedding_dim = text_embedding_size_to_dim[text_embedding_size]
  backbone_config = load_t5_efficient_backbone_config(
      backbone_size=backbone_size
  )
  backbone_embedding_dim = backbone_size_to_dim[backbone_size]

  # TODO(mononito): Move this into a function, but functions can't be used in
  # fiddle configs, easily.
  # Parse z_key_to_val_dim_ratio string into z_key_dim and z_val_dim
  z_key_to_val_dim_ratio = z_key_to_val_dim_ratio.split(":")
  z_key_dim = int(z_key_to_val_dim_ratio[0])
  z_val_dim = int(z_key_to_val_dim_ratio[1])
  z_dim = z_key_dim + z_val_dim
  z_key_dim = int(z_key_dim * backbone_embedding_dim / z_dim)
  z_val_dim = backbone_embedding_dim - z_key_dim

  # Numeric aligner parameters
  n_numeric_bins = 128  # [NOTE] Number of numeric bins
  n_frequencies = 48  # [NOTE] Ignored for simple aligner
  frequency_init_scale = 0.01  # [NOTE] Ignored for simple aligner

  cat_aligner = aligner.TextualAligner(
      x_key_dim=text_embedding_dim,
      x_val_dim=text_embedding_dim,
      z_key_dim=z_key_dim,
      z_val_dim=z_val_dim,
      key_aligner="simple",
      key_bias=False,
      key_activation_fn=None,
      val_aligner="simple",
      val_bias=False,
      val_activation_fn=None,
  )

  str_aligner = aligner.TextualAligner(
      x_key_dim=text_embedding_dim,
      x_val_dim=text_embedding_dim,
      z_key_dim=z_key_dim,
      z_val_dim=z_val_dim,
      key_aligner="simple",
      key_bias=False,
      key_activation_fn=None,
      val_aligner="simple",
      val_bias=False,
      val_activation_fn=None,
  )

  num_aligner = aligner.NumericAligner(
      x_key_dim=text_embedding_dim,
      x_val_dim=n_numeric_bins,
      z_key_dim=z_key_dim,
      z_val_dim=z_val_dim,
      key_aligner="simple",
      key_bias=False,
      key_activation_fn=None,
      val_aligner="simple",
      val_bias=False,
      val_activation_fn=None,
      n_frequencies=n_frequencies,
      frequency_init_scale=frequency_init_scale,
  )

  kv_combiner = kv_combiner_lib.Concatenate()
  kv_combiners = {"num": kv_combiner, "cat": kv_combiner, "str": kv_combiner}

  special_token = generic_embeddings.IngesTablesSpecialTokens(
      z_val_dim=z_val_dim,
  )
  special_tokens = {
      "num": special_token,
      "cat": special_token,
      "str": special_token,
  }

  # NOTE: LLaMA-1 set weight decay to 0.1, while MOMENT used 0.05
  # Earlier this was set to 1e-5
  optimizer = adamw_optimizer(learning_rate=learning_rate, weight_decay=0.1)

  return tabular_foundation_lab.TablularFoundationLab(
      train.Trainer(
          workdir="override_me",
          optimizer=optimizer,
          lr_scheduler=linear_warmup_cosine_lr_scheduler(
              max_steps=num_train_steps,
              warmup_steps=warmup_steps,
              learning_rate=learning_rate,
          ),
          pipeline=pipeline.Pipeline(
              pipeline_modules=[
                  pipeline.PipelineModule(
                      benchmark_name=benchmark_name,
                      dataset_name=dataset_name,
                      splitter=scenario_generators.Splitter(),
                      sampler=scenario_generators.Sampler(),
                      preprocessor=preprocessors.Preprocessor(
                          numeric_scaling_method="quantile",
                          quantile_transformer_method="sklearn",
                      ),
                      encoder=encoders.Encoder(
                          max_num_categories=8,
                          n_bins=128,
                          text_encoder=text_encoders.TextEncoder(
                              text_encoder_name="st5",
                              model_path=ROOT_DIR
                              + f"huggingface/sentence-t5-{text_embedding_size}/",
                              do_lower_case=False,
                              max_seq_length=512,
                              embedding_dim=text_embedding_dim,
                              use_gpu=True,
                              use_fast=True,
                              vocab_size=None,
                              encoding_batch_size=128,
                          ),
                          remove_target_from_feature=False,
                          binning_method="uniform",
                          target_encoding="raw",
                          serialize_columns=False,
                      ),
                  ),
              ],
          ),
          enable_amp=True,
          amp_dtype=torch.bfloat16,  # Only support in A100 GPUs and above
          train_batch_size=128,
          eval_batch_size=512,
          num_train_steps=num_train_steps,
          log_loss_every_steps=10,
          eval_every_steps=20,  # There are approximately 20 datasets.
          checkpoint_every_steps=100,
          num_data_workers=0,
          prefetch_factor=None,
          metrics_writer=default_metrics_writer(),
          compile_model=False,  # Does not work
          masking_strategy=masking.MaskingStrategy(
              target_masking_prob=target_masking_prob,
              default_masking_prob=default_masking_prob,
          ),
          model=ingestables.Model(
              aligners={
                  "cat": cat_aligner,
                  "str": str_aligner,
                  "num": num_aligner,
              },
              kv_combiner=kv_combiners,
              backbone=t5_transformer.T5EncoderModel(config=backbone_config),
              heads={
                  "cat": head.IngesTablesClassification(
                      aligner=cat_aligner,
                      kv_combiner=kv_combiners["cat"],
                      max_num_classes=8,
                  ),
                  "num": head.IngesTablesRegression(
                      z_dim=backbone_embedding_dim,
                  ),
              },
              special_tokens=special_tokens,
          ),
          ingestables_checkpoint_name_and_path=ingestables_checkpoint_name_and_path,
          pretrained_backbone_name_and_path=ROOT_DIR
          + f"huggingface/t5-efficient-{backbone_size}/",
          load_backbone_weights=load_backbone_weights,
          load_aligner_weights_if_available=load_aligner_weights_if_available,
          load_head_weights_if_available=load_head_weights_if_available,
          freeze_backbone=freeze_backbone,
          freeze_aligners=freeze_aligners,
          freeze_heads=freeze_heads,
          freeze_special_tokens=freeze_special_tokens,
      ),
      is_gpu_required=True,
  )


def add_pretraining_pipeline_modules(
    basic_pipeline_module: pipeline.PipelineModule,
    benchmark_name: str = "carte",
    pretraining_dataset_names: Tuple[
        str, ...
    ] = GENERIC_PRETRAINING_DATASET_NAMES,
    text_embedding_size: Literal["base", "large", "xl", "xxl"] = "xl",
    subsample_dataset: bool = False,
    subsampling_type: Literal["examples", "datasets"] = "examples",
    k: float = 1.0,
    subsample_seed: int = 13,
    remove_feature_information: bool = False,
) -> List[pipeline.PipelineModule]:
  """Adds a pipeline object for each pre-training.

  Args:
    basic_pipeline_module: The basic pipeline module to use as a template for
      the pre-training pipeline modules.
    benchmark_name: Name of the benchmark.
    pretraining_dataset_names: The names of the datasets to use for
      pre-training.
    text_embedding_size: The size of the text encoder to use.
    subsample_dataset: Whether to randomly subsample pre-training datasets. This
      is useful for faster pre-training or dataset scaling experiments.
    subsampling_type: Whether to subsample examples or datasets. If "datasets",
      then all examples from a dataset are either included or excluded. If
      "examples", then k% of examples from each dataset are sampled. If
      "datasets", then all examples from k% of datasets are sampled.
    k: k% of each pre-training dataset is randomly sampled.
    subsample_seed: Random state for reproducibility. This controls the sampling
      of the pre-training datasets.
    remove_feature_information: Whether to remove feature information from the
      pre-training data. If yes, then feature name encoding is set to ones.

  Returns:
    A list of pre-training pipeline modules.
  """
  logging.info("Adding pre-training pipeline modules")

  sampler = None
  if subsample_dataset and subsampling_type == "examples":
    logging.info("Subsampling pre-training datasets with k=%f", k)
    sampler = scenario_generators.Sampler(
        sampling_type="random",
        k=k,
        random_state=subsample_seed,
    )
  if subsample_dataset and subsampling_type == "datasets":
    logging.info("Pre-training on k=%f of pre-training datasets", k)
    rng = np.random.default_rng(subsample_seed)
    total_pretraining_datasets = len(pretraining_dataset_names)
    pretraining_dataset_names = rng.choice(
        pretraining_dataset_names,
        size=int(k * total_pretraining_datasets),
        replace=False,
    ).tolist()
    logging.info(
        "Pre-training on %d datasets: %s",
        len(pretraining_dataset_names),
        pretraining_dataset_names,
    )

  pipeline_modules = []
  # Initialize text encoder here so that it is shared across all d atasets.
  text_encoder = text_encoders.TextEncoder(
      text_encoder_name="st5",
      model_path=ROOT_DIR + f"huggingface/sentence-t5-{text_embedding_size}/",
      embedding_dim=text_embedding_size_to_dim[text_embedding_size],
      encoding_batch_size=1024,
  )

  for dataset_name in tqdm.tqdm(
      pretraining_dataset_names, total=len(pretraining_dataset_names)
  ):
    pipeline_module = copy.deepcopy(basic_pipeline_module)
    pipeline_module.benchmark_name = benchmark_name
    pipeline_module.dataset_name = dataset_name
    pipeline_module.encoder.text_encoder = text_encoder
    if subsample_dataset and subsampling_type == "examples":
      pipeline_module.sampler = sampler
    if remove_feature_information:
      pipeline_module.encoder.feature_name_encoding = "ones"
    pipeline_module.splitter = scenario_generators.Splitter(
        random_state=subsample_seed,
    )
    pipeline_modules.append(pipeline_module)

  return pipeline_modules


@auto_config.auto_config
def adamw_optimizer(learning_rate: float = 3e-4, weight_decay: float = 1e-5):
  """Returns a function that binds an optimizer to model parameters.

  This is a partial because parameters are not known at config definition time.

  Args:
    learning_rate: Learning rate
    weight_decay: Weight decay
  """
  return functools.partial(
      torch.optim.AdamW, lr=learning_rate, weight_decay=weight_decay
  )


### Learning rate schedulers
@auto_config.auto_config
def linear_warmup_cosine_lr_scheduler(
    max_steps: int = 1000, warmup_steps: int = 100, learning_rate: float = 3e-4
):
  """Returns a function that binds a learning rate scheduler to an optimizer.

  Set the learning rate of each parameter group using a cosine annealing
  schedule with a linear warmup.

  Args:
    max_steps: The maximum number of steps to train for.
    warmup_steps: The number of steps to warmup the learning rate for.
    learning_rate: The peak learning rate.
  """
  return functools.partial(
      lr_scheduler.LinearWarmupCosineLRScheduler,
      max_steps=max_steps,
      warmup_steps=warmup_steps,
      warmup_start_lr=0.0,
      warmup_end_lr=-1,
      peak_lr=learning_rate,
      min_lr=0.1 * learning_rate,
      simulate_lr_schedule=False,
  )


@auto_config.auto_config
def cosine_annealing_lr():
  """Returns a function that binds a learning rate scheduler to an optimizer.

  Set the learning rate of each parameter group using a cosine annealing
  schedule.
  """
  return functools.partial(
      torch.optim.lr_scheduler.CosineAnnealingLR,
      T_max=10,
      eta_min=1e-5,  # Minimum learning rate
      last_epoch=-1,  # Index of the last epoch
  )


@auto_config.auto_config
def one_cycle_lr():
  """Returns a function that binds a learning rate scheduler to an optimizer.

  Set the learning rate of each parameter group using a cosine annealing
  schedule.
  """
  return functools.partial(
      torch.optim.lr_scheduler.OneCycleLR,
      max_lr=1e-4,  # Maximum learning rate
      total_steps=1e-5,  #  The total number of steps in the cycle.
      last_epoch=-1,  # Index of the last epoch
      anneal_strategy="cos",  # Annealing strategy, one of "cos" and "linear"
      pct_start=0.3,  # The percentage of the cycle spent increasing the LR
  )


@auto_config.auto_config
def exponential_decay_lr():
  """Returns a function that binds a learning rate scheduler to an optimizer.

  Decays the learning rate of each parameter group by gamma every epoch.
  """
  return functools.partial(
      torch.optim.lr_scheduler.ExponentialLR,
      gamma=0.1,  # Multiplicative factor of learning rate decay.
      last_epoch=-1,  # Index of the last epoch. Default: -1.
  )


@auto_config.auto_config
def default_metrics_writer() -> metrics.MetricsWriter:
  """Makes a basic fiddle config for ingestables.Model."""
  return metrics.MetricsWriter(metrics.TensorboardStore("override_me"))


@auto_config.auto_config
def load_t5_efficient_backbone_config(
    backbone_size: Literal["tiny", "mini", "small", "base", "large"] = "tiny",
):
  path = ROOT_DIR + f"huggingface/t5-efficient-{backbone_size}/"
  path = epath.Path(path)
  t5_config = transformers.AutoConfig.from_pretrained(
      path, local_files_only=True
  )
  return t5_config


def set_is_gpu_required(config, value):
  config.is_gpu_required = value
  return config


def ingestables_basic_experiment():
  lab = full_base_config.as_buildable()
  return lab


# Numeric Featurization with Periodic Embeddings
def ingestables_numeric_periodic_featurization_experiment():
  """Numeric featurization with periodic embeddings."""
  lab = full_base_config.as_buildable()
  lab.trainer.pipeline.pipeline_modules[0].encoder.numeric_encoding = "raw"
  lab.trainer.model.aligners["num"] = aligner.NumericAligner(
      x_key_dim=768,  # Assume base text encoder
      x_val_dim=1,  # Only works for raw numeric features
      z_key_dim=64,  # Assume Tiny backbone
      z_val_dim=192,  # Assume Tiny backbone
      key_aligner="simple",
      key_bias=False,
      key_activation_fn=None,
      val_aligner="periodic",
      val_bias=False,
      val_activation_fn=None,
      n_frequencies=48,
      frequency_init_scale=0.01,
  )
  return lab


def ingestables_tp_berta_like_numeric_embedding_experiment():
  """Experiment where IngesTables uses TP-BERTa like numeric embedding.

  In this experiment, the numeric features are embedded by scaling the key
  embeddings using the raw numeric values.

  Returns:
    A TablularFoundationLab with the TP-BERTa like numeric embedding experiment.
  """
  lab = full_base_config.as_buildable()

  lab.trainer.pipeline.pipeline_modules[0].encoder.numeric_encoding = "raw"
  lab.trainer.model.aligners["num"] = aligner.NumericAligner(
      x_key_dim=768,  # Assume base text encoder
      x_val_dim=1,  # Only works for raw numeric features
      z_key_dim=256,  # Assume Tiny backbone
      z_val_dim=1,  # Only works for raw numeric features
      key_aligner="simple",
      key_bias=False,
      key_activation_fn=None,
      val_aligner="identity",
      val_bias=False,
      val_activation_fn=None,
      n_frequencies=48,
      frequency_init_scale=0.01,
  )
  lab.trainer.model.kv_combiner["num"] = kv_combiner_lib.Scale()
  # The numeric special token needs to be 1-D
  lab.trainer.model.special_tokens["num"] = (
      generic_embeddings.IngesTablesSpecialTokens(z_val_dim=1)
  )

  return lab


# Vary sizes of text encoder.
def ingestables_base_text_encoder_experiment():
  lab = full_base_config.as_buildable(text_embedding_size="base")
  return lab


def ingestables_large_text_encoder_experiment():
  lab = full_base_config.as_buildable(text_embedding_size="large")
  return lab


def ingestables_xl_text_encoder_experiment():
  lab = full_base_config.as_buildable(text_embedding_size="xl")
  return lab


# Vary IngesTables backbone
def ingestables_mini_backbone_experiment():
  lab = full_base_config.as_buildable(backbone_size="mini")
  return lab


def ingestables_small_backbone_experiment():
  lab = full_base_config.as_buildable(backbone_size="small")
  return lab


def ingestables_base_backbone_experiment():
  lab = full_base_config.as_buildable(backbone_size="base")
  return lab


def ingestables_large_backbone_experiment():
  lab = full_base_config.as_buildable(backbone_size="large")
  return lab


# Vary Key to Value Dimensions
def ingestables_large_key_val_dim_experiment():
  lab = full_base_config.as_buildable(z_key_to_val_dim_ratio="3:1")
  return lab


def ingestables_equal_key_val_dim_experiment():
  lab = full_base_config.as_buildable(z_key_to_val_dim_ratio="1:1")
  return lab


def ingestables_small_key_val_dim_experiment():
  lab = full_base_config.as_buildable(z_key_to_val_dim_ratio="1:3")
  return lab


### IngesTables Pre-training Experiments
def pretrain_tiny_ingestables():
  """Pretrains a Tiny IngesTables model on generic CARTE pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module,
  )

  return lab


def pretrain_mini_ingestables():
  """Pretrains a Mini IngesTables model on generic CARTE pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="mini",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module
  )

  return lab


def pretrain_small_ingestables():
  """Pretrains a Small IngesTables model on generic CARTE pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="small",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module
  )

  return lab


def pretrain_base_ingestables():
  """Pretrains a Base IngesTables model on generic CARTE pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="base",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module
  )

  return lab


### In-domain Pre-training Experiments
def pretrain_base_ingestables_in_domain_alcohol():
  """Pretrains a Base IngesTables model on Alcohol In-domain pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=4 * 5000,
      backbone_size="base",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module=basic_pipeline_module,
      benchmark_name="carte",
      pretraining_dataset_names=ALCOHOL_PRETRAINING_DATASET_NAMES,
  )

  return lab


def pretrain_base_ingestables_in_domain_credit():
  """Pretrains a Base IngesTables model on Credit In-domain pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=4 * 5000,
      backbone_size="base",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module,
      benchmark_name="ingestables",
      pretraining_dataset_names=CREDIT_PRETRAINING_DATASET_NAMES,
  )

  return lab


def pretrain_base_ingestables_in_domain_housing():
  """Pretrains a Base IngesTables model on Housing In-domain pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=4 * 5000,
      backbone_size="base",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module,
      benchmark_name="ingestables",
      pretraining_dataset_names=HOUSING_PRETRAINING_DATASET_NAMES,
  )

  return lab


def pretrain_base_ingestables_in_domain_restaurant():
  """Pretrains a Base IngesTables model on Housing In-domain pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=2 * 5000,
      backbone_size="base",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module,
      benchmark_name="carte",
      pretraining_dataset_names=RESTAURANT_PRETRAINING_DATASET_NAMES,
  )

  return lab


### Text Encoder Scaling Experiments
def pretrain_tiny_ingestables_large_text_encoder():
  """Pretrains a Tiny IngesTables model on generic CARTE pre-training datasets with a large text encoder."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
      text_embedding_size="large",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module=basic_pipeline_module, text_embedding_size="large"
  )

  return lab


def pretrain_tiny_ingestables_base_text_encoder():
  """Pretrains a Tiny IngesTables model on generic CARTE pre-training datasets with a base text encoder."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
      text_embedding_size="base",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module=basic_pipeline_module, text_embedding_size="base"
  )

  return lab


def pretrain_base_ingestables_large_text_encoder():
  """Pretrains a Base IngesTables model on generic CARTE pre-training datasets with a large text encoder."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="base",
      text_embedding_size="large",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module=basic_pipeline_module, text_embedding_size="large"
  )

  return lab


def pretrain_base_ingestables_base_text_encoder():
  """Pretrains a Base IngesTables model on generic CARTE pre-training datasets with a base text encoder."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="base",
      text_embedding_size="base",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module=basic_pipeline_module, text_embedding_size="base"
  )

  return lab


### Data Scaling Experiments
def pretrain_on_k_percent_data_with_seed_s(k: float, subsample_seed: int):
  """Pretrains a Tiny IngesTables model on k% of randomly sampled all generic pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
      subsample_seed=subsample_seed,
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module=basic_pipeline_module,
      subsample_dataset=True,
      k=k,
      subsampling_type="examples",
  )

  return lab


def pretrain_on_75_percent_data_with_seed_13():
  return pretrain_on_k_percent_data_with_seed_s(k=0.75, subsample_seed=13)


def pretrain_on_75_percent_data_with_seed_14():
  return pretrain_on_k_percent_data_with_seed_s(k=0.75, subsample_seed=14)


def pretrain_on_75_percent_data_with_seed_15():
  return pretrain_on_k_percent_data_with_seed_s(k=0.75, subsample_seed=15)


def pretrain_on_50_percent_data_with_seed_13():
  return pretrain_on_k_percent_data_with_seed_s(k=0.5, subsample_seed=13)


def pretrain_on_50_percent_data_with_seed_14():
  return pretrain_on_k_percent_data_with_seed_s(k=0.5, subsample_seed=14)


def pretrain_on_50_percent_data_with_seed_15():
  return pretrain_on_k_percent_data_with_seed_s(k=0.5, subsample_seed=15)


def pretrain_on_k_percent_datasets_with_seed_s(k: float, subsample_seed: int):
  """Pretrains a Tiny IngesTables model on k% of generic pre-training datasets."""
  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module=basic_pipeline_module,
      subsample_dataset=True,
      k=k,
      subsample_seed=subsample_seed,
      subsampling_type="datasets",
  )

  return lab


def pretrain_on_75_percent_datasets_with_seed_13():
  return pretrain_on_k_percent_datasets_with_seed_s(k=0.75, subsample_seed=13)


def pretrain_on_75_percent_datasets_with_seed_14():
  return pretrain_on_k_percent_datasets_with_seed_s(k=0.75, subsample_seed=14)


def pretrain_on_75_percent_datasets_with_seed_15():
  return pretrain_on_k_percent_datasets_with_seed_s(k=0.75, subsample_seed=15)


def pretrain_on_50_percent_datasets_with_seed_13():
  return pretrain_on_k_percent_datasets_with_seed_s(k=0.5, subsample_seed=13)


def pretrain_on_50_percent_datasets_with_seed_14():
  return pretrain_on_k_percent_datasets_with_seed_s(k=0.5, subsample_seed=14)


def pretrain_on_50_percent_datasets_with_seed_15():
  return pretrain_on_k_percent_datasets_with_seed_s(k=0.5, subsample_seed=15)


# Impact of Quality of Training Data
def impact_of_feature_names_on_pretraining():
  """Impact of Feature Names on Generic pre-training."""

  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module,
      remove_feature_information=True,
      text_embedding_size="xl",
  )

  return lab


def randomly_picked_target_pretraining():
  """Simulates picking targets uniformly at random during pre-training."""

  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
      default_masking_prob=0.3,
      target_masking_prob=0.3,
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module
  )

  return lab


def ignore_target_pretraining():
  """Simulates ignoring targets during pre-training."""

  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
      default_masking_prob=0.3,
      target_masking_prob=0.0,
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module
  )

  return lab


# Fully-supervised vs more-than-supervised training
def only_supervised_pretraining():
  """Fully-supervised pre-training."""

  lab = full_base_config.as_buildable(
      num_train_steps=100000,
      backbone_size="tiny",
      default_masking_prob=0.0,
      target_masking_prob=1.0,
  )

  basic_pipeline_module = lab.trainer.pipeline.pipeline_modules[0]
  lab.trainer.pipeline.pipeline_modules = add_pretraining_pipeline_modules(
      basic_pipeline_module
  )

  return lab


### k-shot fine-tuning IngesTables
def basic_transfer_learning_experiment(
    backbone_size: Literal["tiny", "small", "base"] = "base",
    text_embedding_size: Literal["base", "large", "xl", "xxl"] = "xl",
    pretraining_corpus_name: Literal[
        "none", "generic", "alcohol", "credit", "housing", "restaurant"
    ] = "none",
):
  """Loads a pre-trained IngesTables model.

  Args:
    backbone_size: The size of the backbone to use.
    text_embedding_size: The size of the text encoder to use.
    pretraining_corpus_name: The name of the corpus used for pre-training. If
      "none", no pre-training corpus is used, and the model is randomly
      initialized, with backbone weights loaded from a T5-efficient checkpoint
      (by default)

  Returns:
    A Lab object for transfer learning.
  """
  if pretraining_corpus_name == "none":
    pretrained_model_dir_path = None
  else:
    checkpoint_name = (
        f"{pretraining_corpus_name}-{backbone_size}-{text_embedding_size}"
    )
    pretrained_model_dir_path = PRETRAINED_INGESTABLES_PATHS[checkpoint_name]
    logging.info(
        "Loading IngesTables (%s) checkpoint at %s",
        checkpoint_name,
        pretrained_model_dir_path,
    )

  load_aligner_weights_if_available = True
  # NOTE: Always True to reap the benefits of T5-efficient pre-training.
  load_backbone_weights = True
  load_head_weights_if_available = True

  # Freeze parameters
  freeze_backbone = False
  freeze_aligners = False
  freeze_heads = False
  freeze_special_tokens = False

  if pretrained_model_dir_path is None:
    logging.info("No pretrained model directory path provided.")
    load_aligner_weights_if_available = False
    load_head_weights_if_available = False

  lab = full_base_config.as_buildable(
      backbone_size=backbone_size,
      text_embedding_size=text_embedding_size,
      ingestables_checkpoint_name_and_path=pretrained_model_dir_path,
      load_aligner_weights_if_available=load_aligner_weights_if_available,
      load_backbone_weights=load_backbone_weights,
      load_head_weights_if_available=load_head_weights_if_available,
      freeze_backbone=freeze_backbone,
      freeze_aligners=freeze_aligners,
      freeze_heads=freeze_heads,
      freeze_special_tokens=freeze_special_tokens,
  )
  return lab


def ingestables_base_cross_domain_transfer_experiment():
  """Pre-trained IngesTables Cross-domain transfer learning experiment."""
  return basic_transfer_learning_experiment(
      backbone_size="base",
      text_embedding_size="xl",
      pretraining_corpus_name="generic",
  )


def ingestables_tiny_cross_domain_transfer_experiment():
  """Pre-trained IngesTables Cross-domain transfer learning experiment."""
  return basic_transfer_learning_experiment(
      backbone_size="tiny",
      text_embedding_size="xl",
      pretraining_corpus_name="generic",
  )


################################################################################
###################### DEEP BASELINES CONFIGS ##################################
################################################################################


@auto_config.auto_config
def mlp_config(input_dim, output_dim) -> mlp.MLP:
  """Makes a basic fiddle config for MLP model.

  By default, the model is a regression model.

  Args:
    input_dim: Input dimension.
    output_dim: Output dimension.

  Returns:
    A MLP model.
  """
  return mlp.MLP(
      d_in=input_dim,
      d_out=output_dim,
      task_type="regression",
      head=head.SklearnRegression(),
      n_blocks=2,
      d_block=384,
      dropout=0.1,
  )


@auto_config.auto_config
def resnet_config(input_dim, output_dim) -> resnet.ResNet:
  """Makes a basic fiddle config for ResNet model.

  By default, the model is a regression model.

  Args:
    input_dim: Input dimension.
    output_dim: Output dimension.

  Returns:
    A ResNet model.
  """
  return resnet.ResNet(
      d_in=input_dim,
      d_out=output_dim,
      task_type="regression",
      head=head.SklearnRegression(),
      n_blocks=2,
      d_block=192,
      d_hidden=None,
      d_hidden_multiplier=2.0,
      dropout1=0.3,
      dropout2=0.0,
  )


@auto_config.auto_config
def ft_transformer_config(
    n_cont_features: int, output_dim: int, cat_cardinalities: List[int]
) -> ft_transformer.FTTransformer:
  """Makes a basic fiddle config for FT-Transformer model.

  By default, the model is a regression model.

  Args:
    n_cont_features: Number of continuous features.
    output_dim: Output dimension.
    cat_cardinalities: Cardinalities of categorical features.

  Returns:
    A FT-Transformer model.
  """
  return ft_transformer.FTTransformer(
      n_cont_features=n_cont_features,
      cat_cardinalities=cat_cardinalities,
      d_out=output_dim,
      task_type="regression",
      head=head.SklearnRegression(),
      **ft_transformer.FTTransformer.get_default_kwargs(),
  )


@auto_config.auto_config
def basic_deep_trainer(
    benchmark_name: str = "carte",
    dataset_name: str = "cardekho",
    num_train_steps: int = 10000,
) -> train_deep.Trainer:
  """Makes a basic fiddle config for training MLP and ResNet models."""

  # Learning rate parameters
  learning_rate = 3e-4  # [NOTE] Hyper-parameter
  warmup_steps = 100  # [NOTE] Hyper-parameter

  max_input_dim = 128  # [NOTE] Hyper-parameter, dataset dependent.
  output_dim = 1  # [NOTE] Dataset and task dependent.
  model = mlp_config(input_dim=max_input_dim, output_dim=output_dim)

  return train_deep.Trainer(
      workdir="override_me",
      model=model,
      optimizer=adamw_optimizer(learning_rate=learning_rate, weight_decay=1e-5),
      lr_scheduler=linear_warmup_cosine_lr_scheduler(
          max_steps=num_train_steps,
          warmup_steps=warmup_steps,
          learning_rate=learning_rate,
      ),
      pipeline=pipeline.Pipeline(
          pipeline_modules=[
              pipeline.PipelineModule(
                  benchmark_name=benchmark_name,
                  dataset_name=dataset_name,
                  splitter=scenario_generators.Splitter(),
                  sampler=scenario_generators.Sampler(),
                  preprocessor=preprocessors.Preprocessor(),
                  encoder=encoders.Encoder(
                      max_num_categories=128,
                      target_encoding="label_encoding",
                      categorical_encoding="one_hot",
                      numeric_encoding="raw",
                      feature_name_encoding="none",
                      string_encoding="none",
                      text_encoder=text_encoders.TextEncoder(
                          encoding_batch_size=128, text_encoder_name="stub"
                      ),
                      serialize_columns=False,
                      remove_target_from_feature=True,
                  ),
              ),
          ],
      ),
      train_batch_size=1024,
      eval_batch_size=1024,
      enable_amp=False,
      amp_dtype=None,
      num_train_steps=num_train_steps,
      log_loss_every_steps=10,
      eval_every_steps=10,
      checkpoint_every_steps=100,
      num_data_workers=0,
      prefetch_factor=None,
      metrics_writer=default_metrics_writer(),
      compile_model=False,
  )


@auto_config.auto_config
def basic_experiment():
  trainer = basic_deep_trainer()
  lab = tabular_foundation_lab.TablularFoundationLab(
      trainer=trainer, is_gpu_required=True
  )
  return lab


def mlp_regression_experiment():
  lab = basic_experiment.as_buildable()
  lab.trainer.model = mlp_config(input_dim=128, output_dim=1)
  return lab


def resnet_regression_experiment():
  lab = basic_experiment.as_buildable()
  lab.trainer.model = resnet_config(input_dim=128, output_dim=1)
  return lab


def ft_transformer_regression_experiment():
  """FT-Transformer regression experiment.

  Returns:
    A basic fiddle config for FT-Transformer regression experiment.
  """
  lab = basic_experiment.as_buildable()

  max_num_categories = 48
  n_cat_features = 9  # [NOTE] Dataset dependent.
  n_cont_features = 2  # [NOTE] Dataset dependent.
  output_dim = 1  # [NOTE] Dataset and task dependent. 1 for regression.

  lab.trainer.model = ft_transformer_config(
      n_cont_features=n_cont_features,
      output_dim=output_dim,
      # cat_cardinalities=[12, 3, 6, 5, 12, 2, 34, 6, 2],
      cat_cardinalities=n_cat_features * [max_num_categories],
      # [NOTE] We sort categorical features by name in the encoder.
      # The order of cardinalities should match the order of features.
      # We can set this to the maximum number of categories, so we don't need
      # to change this for every dataset.
  )
  lab.trainer.pipeline.pipeline_modules[0].encoder.categorical_encoding = (
      "ordinal"
  )
  return lab


################################################################################
###################### TABLLM CONFIGS #########################################
################################################################################


@auto_config.auto_config
def tabllm_trainer(
    benchmark_name: str = "carte",
    dataset_name: str = "cardekho",
) -> train_sklearn.Trainer:
  """Makes a basic fiddle config for training tabllm models."""

  model = sklearn_random_forest_regressor_config()

  return train_sklearn.Trainer(
      workdir="override_me",
      model=model,
      pipeline=pipeline.Pipeline(
          pipeline_modules=[
              pipeline.PipelineModule(
                  benchmark_name=benchmark_name,
                  dataset_name=dataset_name,
                  splitter=scenario_generators.Splitter(),
                  sampler=scenario_generators.Sampler(),
                  preprocessor=preprocessors.Preprocessor(),
                  encoder=encoders.Encoder(
                      max_num_categories=128,
                      target_encoding="label_encoding",
                      feature_name_encoding="none",
                      string_encoding="llm",
                      text_encoder=text_encoders.TextEncoder(
                          encoding_batch_size=128, text_encoder_name="stub"
                      ),
                      serialize_columns=True,
                      feature_serializer=serialization.FeatureSerializer(),
                      remove_target_from_feature=True,
                  ),
              ),
          ],
      ),
      train_tabllm=True,
      metrics_writer=default_metrics_writer(),
  )


@auto_config.auto_config
def tabllm_experiment_o():
  trainer = tabllm_trainer(benchmark_name="carte", dataset_name="cardekho")
  lab = tabular_foundation_lab.TablularFoundationLab(
      trainer=trainer,
  )
  return lab


def tabllm_random_forest_classification_experiment():
  lab = tabllm_experiment_o.as_buildable()
  lab.trainer.model = sklearn_random_forest_classifier_config()
  return lab


def tabllm_random_forest_regression_experiment():
  lab = tabllm_experiment_o.as_buildable()
  lab.trainer.model = sklearn_random_forest_regressor_config()
  return lab


################################################################################
###################### SKLEARN CONFIGS #########################################
################################################################################


@auto_config.auto_config
def sklearn_trainer(
    model: sklearn_model.SklearnModel,
    benchmark_name: str = "carte",
    dataset_name: str = "cardekho",
) -> train_sklearn.Trainer:
  """Makes a basic fiddle config for training sklearn models."""

  return train_sklearn.Trainer(
      workdir="override_me",
      model=model,
      pipeline=pipeline.Pipeline(
          pipeline_modules=[
              pipeline.PipelineModule(
                  benchmark_name=benchmark_name,
                  dataset_name=dataset_name,
                  splitter=scenario_generators.Splitter(),
                  sampler=scenario_generators.Sampler(),
                  preprocessor=preprocessors.Preprocessor(),
                  encoder=encoders.Encoder(
                      max_num_categories=128,
                      text_encoder=text_encoders.TextEncoder(
                          text_encoder_name="st5"
                      ),
                      target_encoding="label_encoding",
                      numeric_encoding="raw",
                      categorical_encoding="one_hot",
                      string_encoding="none",
                      feature_name_encoding="none",
                      remove_target_from_feature=True,
                  ),
              ),
          ],
      ),
      metrics_writer=default_metrics_writer(),
  )


@auto_config.auto_config
def sklearn_experiment(
    model: sklearn_model.SklearnModel,
    benchmark_name: str = "carte",
    dataset_name: str = "cardekho",
):
  """Makes a basic fiddle config for training sklearn models."""
  trainer = sklearn_trainer(
      model=model,
      benchmark_name=benchmark_name,
      dataset_name=dataset_name,
  )
  lab = tabular_foundation_lab.TablularFoundationLab(
      trainer=trainer,
  )
  return lab


@auto_config.auto_config
def random_forest_classification_experiment():
  return sklearn_experiment(
      model=sklearn_random_forest_classifier_config(),
      benchmark_name="carte",
      dataset_name="michelin",
  )


@auto_config.auto_config
def random_forest_regression_experiment():
  return sklearn_experiment(model=sklearn_random_forest_regressor_config())


@auto_config.auto_config
def lightgbm_classification_experiment():
  return sklearn_experiment(model=sklearn_lightgbm_classifier_config())


@auto_config.auto_config
def lightgbm_regression_experiment():
  return sklearn_experiment(model=sklearn_lightgbm_regressor_config())


@auto_config.auto_config
def logistic_regression_experiment():
  return sklearn_experiment(model=sklearn_logistic_regressor_config())


@auto_config.auto_config
def ridge_regression_experiment():
  return sklearn_experiment(model=sklearn_ridge_regressor_config())


@auto_config.auto_config
def xgboost_classification_experiment():
  return sklearn_experiment(model=xgboost_classifier_config())


@auto_config.auto_config
def xgboost_regression_experiment():
  return sklearn_experiment(model=xgboost_regressor_config())


### Models

# [NOTE]: Defaults are taken from SkLearn or XGBoost documentation.
# [NOTE]: Seeds are set to -1 to ensure it is overridden by the experiment.
# [NOTE]: Specified parameters are only a subset of available parameters. They
# from the CARTE paper for potential hyper-parameter tuning.


### Regression models


@auto_config.auto_config
def sklearn_ridge_regressor_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for Ridge Regression model."""
  return sklearn_model.SklearnModel(
      linear_model.Ridge(
          alpha=1.0,
          solver="auto",
      ),
      head.SklearnRegression(),
  )


@auto_config.auto_config
def xgboost_regressor_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for XGBoost Regression model."""
  return sklearn_model.SklearnModel(
      xgboost.XGBRegressor(  # pylint: disable=attribute-error
          n_estimators=100,
          max_depth=6,
          learning_rate=0.3,
          min_child_weight=1,
          subsample=1,
          colsample_bytree=1,
          col_sample_bylevel=1,
          gamma=0,
          reg_lambda=1,
          reg_alpha=1,
      ),
      head.SklearnRegression(),
  )


@auto_config.auto_config
def sklearn_random_forest_regressor_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for Random Forest regression model."""
  return sklearn_model.SklearnModel(
      ensemble.RandomForestRegressor(
          n_estimators=100,
          max_depth=None,
          max_leaf_nodes=None,
          min_samples_leaf=1,
          bootstrap=True,
          min_impurity_decrease=0.0,
      ),
      head.SklearnRegression(),
  )


@auto_config.auto_config
def sklearn_lightgbm_regressor_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for LightGBM regression model."""
  return sklearn_model.SklearnModel(
      ensemble.HistGradientBoostingRegressor(
          learning_rate=0.1,
          max_depth=None,
          max_leaf_nodes=31,
          min_samples_leaf=20,
          l2_regularization=0.0,
      ),
      head.SklearnRegression(),
  )


### Classification models


@auto_config.auto_config
def xgboost_classifier_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for XGBoost classification model."""
  return sklearn_model.SklearnModel(
      xgboost.XGBClassifier(  # pylint: disable=attribute-error
          n_estimators=100,
          max_depth=6,
          learning_rate=0.3,
          min_child_weight=1,
          subsample=1,
          colsample_bytree=1,
          colsample_bylevel=1,
          gamma=0,
          reg_lambda=1,
          reg_alpha=1,
      ),
      head.SklearnClassification(),
  )


@auto_config.auto_config
def sklearn_logistic_regressor_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for Logistic Regression model."""
  return sklearn_model.SklearnModel(
      linear_model.LogisticRegression(
          penalty="l2",
          solver="lbfgs",
          C=1.0,
      ),
      head.SklearnClassification(),
  )


@auto_config.auto_config
def sklearn_random_forest_classifier_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for Random Forest classification model."""
  return sklearn_model.SklearnModel(
      ensemble.RandomForestClassifier(
          n_estimators=100,
          max_depth=None,
          max_leaf_nodes=None,
          min_samples_leaf=1,
          bootstrap=True,
          min_impurity_decrease=0.0,
      ),
      head.SklearnClassification(),
  )


@auto_config.auto_config
def sklearn_lightgbm_classifier_config() -> sklearn_model.SklearnModel:
  """Makes a basic fiddle config for LightGBM classification model."""
  return sklearn_model.SklearnModel(
      ensemble.HistGradientBoostingClassifier(
          learning_rate=0.1,
          max_depth=None,
          max_leaf_nodes=31,
          min_samples_leaf=20,
          l2_regularization=0.0,
      ),
      head.SklearnClassification(),
  )
