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

"""Trainer lib."""

import itertools
import os
from typing import Any, Callable, Dict, Literal, Tuple

from absl import logging
from etils import epath
from etils import etree
from ingestables.torch import utils
from ingestables.torch.data import encoders
from ingestables.torch.data import pipeline as pipeline_lib
from ingestables.torch.google.baselines import ft_transformer
from ingestables.torch.google.baselines import mlp
from ingestables.torch.google.baselines import resnet
from ingestables.torch.train import metrics
import torch
import torch.distributed as dist  # pylint:disable=g-importing-member
from torch.nn.parallel import DistributedDataParallel as DDP  # pylint:disable=g-importing-member

DeepModelType = mlp.MLP | resnet.ResNet | ft_transformer.FTTransformer


class Trainer:
  """Manages the training loop.

  'optimizer' is a function that takes the model parameters and returns an
  instance of an optimizer.  This function is called 'optimizer' in order for
  the configuration key to be 'optimizer' in the fiddle config.

  Example:
    trainer = Trainer(...)
    trainer.run()
  """

  def __init__(
      self,
      workdir: str,
      model: DeepModelType,
      optimizer: utils.OptimizerBinderFn,
      lr_scheduler: utils.LearningRateSchedulerBinderFn,
      pipeline: pipeline_lib.Pipeline,
      train_batch_size: int,
      eval_batch_size: int,
      num_train_steps: int,
      log_loss_every_steps: int,
      eval_every_steps: int,
      checkpoint_every_steps: int,
      num_data_workers: int = 0,
      prefetch_factor: int | None = 2,
      amp_dtype: torch.dtype | None = None,
      enable_amp: bool = True,
      metrics_writer: metrics.MetricsWriter | None = None,
      compile_model: bool = True,
  ):
    if metrics_writer is None:
      raise ValueError("metrics_writer must be set.")
    self._workdir = epath.Path(workdir)
    self._pipeline = pipeline
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    num_worker_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    node_rank = int(os.environ.get("GROUP_RANK", 0))
    self._world_size = int(os.environ.get("WORLD_SIZE", 1))
    logging.info(
        "local_rank=%s, num_worker_per_node=%s, node_rank=%s, world_size=%s, ",
        local_rank,
        num_worker_per_node,
        node_rank,
        self._world_size,
    )

    gpu_ok = False  # whether the GPU can benefit from compilation.
    if torch.cuda.device_count() > 0:
      self._device = torch.device(f"cuda:{local_rank}")
      torch.cuda.set_device(self._device)
      device_cap = torch.cuda.get_device_capability()
      # Only models that run on V100, A100, or H100 benefit from compilation.
      if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True
    else:
      self._device = torch.device("cpu")
    logging.info("Using device: %s", self._device)

    if self._world_size > 1:
      self._global_rank = node_rank * num_worker_per_node + local_rank
      dist.init_process_group(
          backend="NCCL",
          rank=self._global_rank,
          world_size=self._world_size,
      )
    else:
      self._global_rank = 0

    model = model.to(self._device)
    if gpu_ok:
      model.compile(dynamic=False, disable=not compile_model)
    if self._world_size > 1:
      model = DDP(
          model,
          device_ids=[local_rank],
          output_device=local_rank,
      )
    self._model = model
    self._max_input_dim = model.input_dim

    if isinstance(self._model, ft_transformer.FTTransformer):
      self._optimizer = optimizer(model.make_parameter_groups())
    else:
      self._optimizer = optimizer(model.parameters())

    self._lr_scheduler = lr_scheduler(self._optimizer)
    self._num_train_steps = num_train_steps
    self._log_loss_every_steps = log_loss_every_steps
    self._eval_every_steps = eval_every_steps
    self._checkpoint_every_steps = checkpoint_every_steps
    self._amp_dtype = amp_dtype
    self._enable_amp = enable_amp

    # Dataloader parameters
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size
    self._num_data_workers = num_data_workers
    self._prefetch_factor = prefetch_factor

    self._scaler = torch.amp.GradScaler(device=self._device.type)
    self._train_step_count = 0

    if len(self._pipeline.dataset_keys) > 1:
      raise ValueError(
          "MLP and ResNet models only support single dataset training and"
          " evaluation."
      )
    self.dataset_name = self._pipeline.dataset_keys[0]

    # Create train, val and test dataloaders
    self._train_dataloader = self._get_train_dataloader(self.dataset_name)
    self._val_dataloader = self._get_val_dataloader(self.dataset_name)
    self._test_dataloader = self._get_test_dataloader(self.dataset_name)

    # Train iterators are cycled.
    self._train_iter = iter(itertools.cycle(self._train_dataloader))

    if isinstance(
        self._model, (mlp.MLP, resnet.ResNet, ft_transformer.FTTransformer)
    ):
      self._model_module = self._model
    elif isinstance(self._model, DDP):
      self._model_module = self._model.module
    else:
      raise ValueError(
          "model must be either ingestables.Model or DDP(ingestable.Model). "
          f"Was: {type(self._model)}"
      )

    self._all_metrics = None
    self._metrics_writer = metrics_writer

  def __repr__(self):
    return (
        f"Trainer(workdir={self._workdir},"
        + f" model={self._model},"
        + f" optimizer={self._optimizer},"
        + f" lr_scheduler={self._lr_scheduler},"
        + f" pipeline={self._pipeline},"
        + f" num_train_steps={self._num_train_steps},"
        + f" log_loss_every_steps={self._log_loss_every_steps},"
        + f" eval_every_steps={self._eval_every_steps},"
        + f" checkpoint_every_steps={self._checkpoint_every_steps},"
        + f" amp_dtype={self._amp_dtype},"
        + f" enable_amp={self._enable_amp},"
        + f" metrics_writer={self._metrics_writer}),"
    )

  def _get_dataloader(
      self,
      dataset: pipeline_lib.IndexedDataset,
      batch_size: int = 128,
      shuffle: bool = False,
      collate_fn: (
          Callable[[Any], Any] | None
      ) = utils.preprocessed_inputs_collate_fn,
      drop_last: bool = True,
  ) -> torch.utils.data.DataLoader:
    """Returns the data loader for the given dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
        num_workers=self._num_data_workers,
        prefetch_factor=self._prefetch_factor,
    )

  def _prepare_inputs_basic(
      self, cat: torch.Tensor | None, num: torch.Tensor | None
  ) -> Dict[str, torch.Tensor]:
    """Formats encoded data into input features and targets."""

    # Concatenate features.
    if cat is None:
      features = num
    elif num is None:
      features = cat
    elif cat is not None and num is not None:
      features = torch.cat([cat, num], dim=-1)
    else:
      raise ValueError("No features to train on.")

    # Pad features to max input dimension.
    if features is not None:

      if features.shape[-1] > self._max_input_dim:
        raise ValueError(
            f"Features with {features.shape[-1]=} have more dimensions than max"
            f" input dimension {self._max_input_dim}."
        )

      padding = self._max_input_dim - features.shape[-1]
      if padding > 0:
        features = torch.nn.functional.pad(
            features, (0, padding, 0, 0), mode="constant", value=0
        )

      features = features.float()

    return {"x": features}

  def _prepare_inputs_ft_transformer(
      self, cat: torch.Tensor | None, num: torch.Tensor | None
  ) -> Dict[str, torch.Tensor]:
    """Formats encoded data into input features and targets."""
    if cat is None and num is None:
      raise ValueError("No features to train on.")
    else:
      return {
          "x_cont": num.float() if num is not None else None,
          "x_cat": cat.long() if cat is not None else None,
      }

  def _format_inputs(
      self, data_: encoders.PreprocessedInputs
  ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Formats encoded data into input features and targets."""
    cat = (
        None
        if data_.encoded_categorical is None
        else data_.encoded_categorical.float()
    )
    cat = cat.squeeze(-1) if cat is not None and cat.ndim == 3 else cat

    num = (
        None if data_.encoded_numeric is None else data_.encoded_numeric.float()
    )
    num = num.squeeze(-1) if num is not None and num.ndim == 3 else num

    target = data_.encoded_targets

    # TODO(mononito): Maybe the prepare inputs should be implemented in the
    # model?
    if isinstance(self._model, ft_transformer.FTTransformer):
      features = self._prepare_inputs_ft_transformer(cat, num)
    elif isinstance(self._model, (mlp.MLP, resnet.ResNet)):
      features = self._prepare_inputs_basic(cat, num)
    else:
      raise ValueError(
          "Unsupported model type for _format_inputs: "
          f"{type(self._model).__name__}"
      )

    return features, target

  def _get_train_dataloader(
      self,
      dataset_key: str,
  ) -> torch.utils.data.DataLoader:
    """Returns the train data loader for the given dataset."""
    return self._get_dataloader(
        self._pipeline.get_train_data(dataset_key),
        batch_size=self._train_batch_size,
        shuffle=True,
        collate_fn=utils.preprocessed_inputs_collate_fn,
        drop_last=True,  # drop last batch
    )

  def _get_test_dataloader(
      self,
      dataset_key: str,
  ) -> torch.utils.data.DataLoader:
    """Returns the test data loader for the given dataset."""
    return self._get_dataloader(
        self._pipeline.get_test_data(dataset_key),
        batch_size=self._eval_batch_size,
        shuffle=False,
        collate_fn=utils.preprocessed_inputs_collate_fn,
        drop_last=False,  # no need to keep last batch
    )

  def _get_val_dataloader(
      self,
      dataset_key: str,
  ) -> torch.utils.data.DataLoader:
    """Returns the val data loader for the given dataset."""
    return self._get_dataloader(
        self._pipeline.get_val_data(dataset_key),
        batch_size=self._eval_batch_size,
        shuffle=False,
        collate_fn=utils.preprocessed_inputs_collate_fn,
        drop_last=False,  # no need to keep last batch
    )

  def _train_step(
      self, dataset_key: str, batch: encoders.PreprocessedInputs
  ) -> None:
    """Run a single training step."""
    self._model.train()

    task_info = self._pipeline.get_task_info(dataset_key)
    logging.info(
        "Training step %s on dataset: %s", self._train_step_count, dataset_key
    )
    inputs, targets = self._format_inputs(batch)
    inputs = etree.map(lambda x: x.to(self._device), inputs)

    targets = (
        targets.long()
        if task_info.task_type == "classification"
        else targets.float()
    )
    targets = targets.to(self._device)

    self._optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=self._device.type,
        dtype=self._amp_dtype,
        enabled=self._enable_amp,
    ):
      logits = self._model(**inputs)

    loss = self._model.loss(logits=logits, y_true=targets)
    self._scaler.scale(loss).backward()
    self._scaler.unscale_(self._optimizer)
    # NOTE: LLaMA-1 used gradient clipping 1.0, wherease MOMENT used 5.0.
    # Earlier max_norm was set to 10.0
    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
    self._scaler.step(self._optimizer)
    self._scaler.update()

    loss = loss.item()
    self._metrics_writer.write_metric(
        self._train_step_count,
        ("Sum",),
        "loss",
        loss,
    )

  def _evaluate(
      self,
      dataset_type: Literal["test", "val"],
      eval_dataloader: torch.utils.data.DataLoader,
  ) -> None:
    """Evaluates the model on test datasets."""
    self._model.eval()

    task_info = self._pipeline.get_task_info(self.dataset_name)
    logging.info(
        "Evaluating on `%s` dataset on the `%s` task",
        self.dataset_name,
        task_info.task_type,
    )

    batched_logits = []
    batched_targets = []
    for batch in eval_dataloader:
      inputs, targets = self._format_inputs(batch)
      inputs = etree.map(lambda x: x.to(self._device), inputs)
      targets = (
          targets.long()
          if task_info.task_type == "classification"
          else targets.float()
      )
      targets = targets.to(self._device)

      with torch.no_grad():
        with torch.autocast(
            device_type=self._device.type,
            dtype=self._amp_dtype,
            enabled=self._enable_amp,
        ):
          logits = self._model(**inputs)

      # Convert to float as logits may be of type torch.bfloat16.
      logits = logits.float()

      batched_logits.append(logits)
      batched_targets.append(targets)

    batched_logits = torch.cat(batched_logits, dim=0)
    batched_targets = torch.cat(batched_targets, dim=0)

    metrics_ = self._model.compute_metrics(
        y_true=batched_targets, y_probs=batched_logits
    )

    self._metrics_writer.write_model_metrics(
        self._train_step_count,
        dataset_type,
        self.dataset_name,
        metrics_,
    )
    logging.info(
        "Dataset: %s %s, Metrics: %s",
        dataset_type,
        self.dataset_name,
        metrics_,
    )

  def _save_checkpoint(self) -> None:
    if self._global_rank == 0:
      logging.info("Saving model on rank 0 to: %s", self._workdir)
      utils.save_model(
          self._train_step_count,
          self._model,
          self._optimizer,
          self._scaler,
          self._workdir,
      )
    if self._world_size > 1:
      dist.barrier()

  def _get_training_batch(
      self, training_step: int
  ) -> Tuple[str, encoders.PreprocessedInputs]:
    del training_step
    return self.dataset_name, next(self._train_iter)

  def run(self) -> None:
    """Train, evaluate, and save checkpoints."""
    for step in range(self._num_train_steps):
      self._train_step_count = step
      is_last_step = step >= self._num_train_steps - 1

      dataset_key, batch = self._get_training_batch(step)
      self._train_step(dataset_key, batch)
      self._lr_scheduler.step()

      if step % self._eval_every_steps == 0 or is_last_step:
        self._evaluate("test", self._test_dataloader)
        self._evaluate("val", self._val_dataloader)

      if step % self._checkpoint_every_steps == 0 or is_last_step:
        self._save_checkpoint()

    logging.info("Job finished")

  @property
  def eval_metrics(self) -> dict[str, dict[str, Any]]:
    if self._all_metrics is None:
      raise ValueError("No eval metrics. Make sure to call .run() first.")
    return self._all_metrics
