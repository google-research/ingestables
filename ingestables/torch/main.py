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

r"""ENTRYPOINT for running MNIST on PyTorch with GPUs.

Usage:
  python3 -m ingestables.torch.main --output_dir=/tmp/torch_test_00
"""

import os

from absl import app
from absl import flags
from absl import logging
from ingestables.torch.data import dummy_dataset
from ingestables.torch.model import aligner
from ingestables.torch.model import head
from ingestables.torch.model import ingestables
from ingestables.torch.model import kv_combiner
from ingestables.torch.model.backbones import vanilla_transformer
from ingestables.torch.train import train
import torch


_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "output dir", required=True
)

_TRAIN_STEPS = flags.DEFINE_integer("train_steps", 1000, "train steps")
_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    "train_batch_size", 128, "train batch size"
)
_EVAL_BATCH_SIZE = flags.DEFINE_integer("eval_batch_size", 4, "eval batch size")
_LOG_LOSS_EVERY_STEPS = flags.DEFINE_integer(
    "log_loss_every_steps", 100, "every this many steps, log the training loss."
)
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "learning rate")
_MOMENTUM = flags.DEFINE_float("momentum", 0.0, "momentum")
_SEED = flags.DEFINE_integer("seed", 1, "seed")
_NUM_DATA_WORKERS = flags.DEFINE_integer(
    "num_data_workers", 1, "number of processes used by the train data loader."
)

_IS_GPU_REQUIRED = flags.DEFINE_boolean(
    "is_gpu_required", False, "fail if not run on GPU"
)


def check_gpu() -> str:
  """Print GPU info and return "cuda" if found, "cpu" otherwise."""
  try:
    logging.info("FLAGS.is_gpu_required: %s", _IS_GPU_REQUIRED.value)
    logging.info("torch.__version__: %s", torch.__version__)
    logging.info("torch.cuda.device_count(): %s", torch.cuda.device_count())
    logging.info("torch.cuda.current_device(): %s", torch.cuda.current_device())
    logging.info(
        "torch.cuda.get_device_name(0): %s", torch.cuda.get_device_name(0)
    )
    logging.info("torch.cuda.is_available(0): %s", torch.cuda.is_available())
    if torch.cuda.is_available():
      return "cuda"
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(e)
  if _IS_GPU_REQUIRED.value:
    logging.error("GPU required, but no GPU found.")
    exit(1)
  logging.error("Falling back to CPU.")
  return "cpu"


def get_model() -> ingestables.Model:
  """Build an ingestables.Model."""
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
  return ingestables.model_from_config(
      aligner_cls_dict=aligner_cls_dict,
      kv_combiner_cls_dict=kv_combiner_cls_dict,
      backbone_cls_dict=backbone_cls_dict,
      head_cls_dict=head_cls_dict,
      config=model_config,
  )


def main(argv):
  del argv
  logging.info("Job started")
  check_gpu()

  logging.info("FLAGS.train_steps: %s", _TRAIN_STEPS.value)
  logging.info("FLAGS.train_batch_size: %s", _TRAIN_BATCH_SIZE.value)
  logging.info("FLAGS.eval_batch_size: %s", _EVAL_BATCH_SIZE.value)
  logging.info("FLAGS.log_loss_every_steps: %s", _LOG_LOSS_EVERY_STEPS.value)
  logging.info("FLAGS.learning_rate: %s", _LEARNING_RATE.value)
  logging.info("FLAGS.momentum: %s", _MOMENTUM.value)
  logging.info("FLAGS.num_data_workers: %s", _NUM_DATA_WORKERS.value)
  local_rank = int(os.environ["LOCAL_RANK"])
  num_worker_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
  node_rank = int(os.environ["GROUP_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])

  train.train_and_test(
      model=get_model(),
      train_datasets={
          "xor": dummy_dataset.XorDataset(),
          "and": dummy_dataset.AndDataset(),
      },
      test_datasets={
          "xor": dummy_dataset.XorDataset(),
          "and": dummy_dataset.AndDataset(),
      },
      train_steps=_TRAIN_STEPS.value,
      train_batch_size=_TRAIN_BATCH_SIZE.value,
      eval_batch_size=_EVAL_BATCH_SIZE.value,
      learning_rate=_LEARNING_RATE.value,
      momentum=_MOMENTUM.value,
      log_loss_every_steps=_LOG_LOSS_EVERY_STEPS.value,
      workdir=_OUTPUT_DIR.value,
      local_rank=local_rank,
      nproc_per_node=num_worker_per_node,
      node_rank=node_rank,
      world_size=world_size,
      num_data_workers=_NUM_DATA_WORKERS.value,
      seed=_SEED.value,
  )


if __name__ == "__main__":
  app.run(main)
