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

# from absl import flags
from absl import logging
from etils import epath
from ingestables.torch import types
from ingestables.torch import utils
from ingestables.torch.data import encoders
from ingestables.torch.data import pipeline as pipeline_lib
from ingestables.torch.model import ingestables
from ingestables.torch.model.backbones import t5_transformer  # pylint: disable=g-importing-member
from ingestables.torch.model.lib import masking
from ingestables.torch.train import metrics
import torch
import torch.distributed as dist  # pylint:disable=g-importing-member
from torch.nn.parallel import DistributedDataParallel as DDP  # pylint:disable=g-importing-member


# # Internal flag for debugging of input shapes.
# _DEBUG_ASSERT_INPUT_SHAPES = flags.DEFINE_bool(
#     "debug_assert_input_shapes",
#     False,
#     "When true, assert input shapes are correct.",
# )


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
      model: ingestables.Model,
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
      masking_strategy: masking.MaskingStrategy | None = None,
      ingestables_checkpoint_name_and_path: str | None = None,
      pretrained_backbone_name_and_path: str | None = None,
      load_backbone_weights: bool = True,
      load_aligner_weights_if_available: bool = False,
      load_head_weights_if_available: bool = False,
      freeze_backbone: bool = False,
      freeze_aligners: bool = False,
      freeze_heads: bool = False,
      freeze_special_tokens: bool = False,
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

    # Load pre-trained weights if available.
    if (
        (ingestables_checkpoint_name_and_path is not None)
        and load_aligner_weights_if_available
        and load_head_weights_if_available
        and load_backbone_weights
    ):
      # Load pre-trained weights from an IngesTables checkpoint.
      # NOTE: ingestables_checkpoint_name_and_path is actually path to the
      # checkpoint directory.
      # TODO(mononito): Change name of the argument to
      # ingestables_checkpoint_dir_path.
      ingestables_checkpoint_name_and_path = self._find_latest_checkpoint_path(
          ingestables_checkpoint_name_and_path
      )
      model.load_state_dict(
          torch.load(ingestables_checkpoint_name_and_path, weights_only=True)
      )
      logging.info(
          "Loaded pre-trained weights from an IngesTables checkpoint: %s",
          ingestables_checkpoint_name_and_path,
      )
    elif (
        load_backbone_weights
        and isinstance(model.encoder.backbone, t5_transformer.T5EncoderModel)
        and (
            not load_aligner_weights_if_available
            or not load_head_weights_if_available
        )
    ):
      # Only load pre-trained weights of the backbone from a T5-efficient
      # checkpoints.
      model.load_pretrained_weights(
          ingestables_checkpoint_name_and_path=ingestables_checkpoint_name_and_path,
          pretrained_backbone_name_and_path=pretrained_backbone_name_and_path,
          load_backbone_weights=load_backbone_weights,
          load_aligner_weights_if_available=load_aligner_weights_if_available,
          load_head_weights_if_available=load_head_weights_if_available,
      )

    # Freeze parameters of the model.
    # TODO(mononito): Implement gradual unfreezing based on train step count
    model.freeze_parameters(
        freeze_backbone=freeze_backbone,
        freeze_aligners=freeze_aligners,
        freeze_heads=freeze_heads,
        freeze_special_tokens=freeze_special_tokens,
    )

    self._ingestables_checkpoint_name_and_path = (
        ingestables_checkpoint_name_and_path
    )
    self._pretrained_backbone_name_and_path = pretrained_backbone_name_and_path
    self._load_backbone_weights = load_backbone_weights
    self._load_aligner_weights_if_available = load_aligner_weights_if_available
    self._load_head_weights_if_available = load_head_weights_if_available
    self._freeze_backbone = freeze_backbone
    self._freeze_aligners = freeze_aligners
    self._freeze_heads = freeze_heads
    self._freeze_special_tokens = freeze_special_tokens

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

    # Training and Evaluation Masking strategy
    self._train_masking_strategy = masking_strategy
    self._eval_masking_strategy = masking.MaskingStrategy(
        target_masking_prob=1.0, default_masking_prob=0.0
    )  # Targets are always masked during eval.

    self._scaler = torch.amp.GradScaler(device=self._device.type)
    self._train_step_count = 0

    # Create train, val and test dataloaders
    self._train_dataloaders = {}
    self._val_dataloaders = {}
    self._test_dataloaders = {}

    # TODO(mononito): Currently the pipeline assumes that the same datasets
    # are used for train/val/test.
    for dataset_key in self._pipeline.dataset_keys:
      self._train_dataloaders[dataset_key] = self._get_train_dataloader(
          dataset_key
      )
      self._val_dataloaders[dataset_key] = self._get_val_dataloader(dataset_key)
      self._test_dataloaders[dataset_key] = self._get_test_dataloader(
          dataset_key
      )

    self._dataset_key_list = list(self._train_dataloaders.keys())
    self._num_datasets = len(self._dataset_key_list)

    # Train iterators are cycled.
    self._train_iters = [
        iter(itertools.cycle(train_loader))
        for train_loader in self._train_dataloaders.values()
    ]

    if isinstance(self._model, ingestables.Model):
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
        + f" ingestables_checkpoint_name_and_path={self._ingestables_checkpoint_name_and_path},"
        + f" pretrained_backbone_name_and_path={self._pretrained_backbone_name_and_path},"
        + f" load_backbone_weights={self._load_backbone_weights},"
        + f" load_aligner_weights_if_available={self._load_aligner_weights_if_available},"
        + f" load_head_weights_if_available={self._load_head_weights_if_available},"
        + f" freeze_backbone={self._freeze_backbone},"
        + f" freeze_aligners={self._freeze_aligners},"
        + f" freeze_heads={self._freeze_heads},"
        + f" freeze_special_tokens={self._freeze_special_tokens},"
    )

  def _find_latest_checkpoint_path(
      self, checkpoint_dir_path: str | epath.Path
  ) -> epath.Path:
    """Finds the latest checkpoint path in the given directory."""
    checkpoint_dir_path = epath.Path(checkpoint_dir_path)
    model_checkpoint_steps = sorted(
        [int(p.name[6:-3]) for p in checkpoint_dir_path.glob("*.pt")]
    )
    total_checkpoints = len(model_checkpoint_steps)
    latest_checkpoint_step = model_checkpoint_steps[-1]
    logging.info(
        "Latest checkpoint is from step {%d}. There are {%d} checkpoints in"
        " total.",
        latest_checkpoint_step,
        total_checkpoints,
    )
    return checkpoint_dir_path / f"model_{latest_checkpoint_step}.pt"

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

  def _format_inputs(self, data_: encoders.PreprocessedInputs) -> Tuple[
      dict[str, types.IngesTablesInferenceInputs | None],
      dict[str, types.IngesTablesTrainingInputs | None],
  ]:
    """Formats encoded data into inference and training inputs."""

    # Infer batch size.
    if data_.encoded_categorical is not None:
      batch_size = data_.encoded_categorical.shape[0]
    elif data_.encoded_numeric is not None:
      batch_size = data_.encoded_numeric.shape[0]
    elif data_.encoded_string is not None:
      batch_size = data_.encoded_string.shape[0]
    else:
      raise ValueError("No data to format.")

    # Infer text encoder dimensions & number of numeric bins
    # n_dims = self._pipeline.get_text_encoder_n_dims(dataset_key)
    # n_bins = self._pipeline.get_numeric_n_bins(dataset_key)

    ### Categorical features. ##################################################
    if data_.encoded_feature_names.categorical is None:
      n_cat_features = 0
    else:
      n_cat_features = len(data_.encoded_feature_names.categorical)

    if n_cat_features == 0:
      cat_inference_inputs = None
      cat_training_inputs = None
    else:
      if data_.encoded_feature_names.categorical is None:
        cat_key_emb = None
      else:
        cat_key_emb = torch.repeat_interleave(
            data_.encoded_feature_names.categorical.unsqueeze(0),
            repeats=batch_size,
            dim=0,
        ).float()
      cat_val_emb = (
          None
          if data_.encoded_categorical is None
          else data_.encoded_categorical.float()
      )
      cat_missing = torch.zeros(
          (batch_size, n_cat_features, 1)
      ).bool()  # [NOTE]: Current implementation does not support missing values
      cat_mask = torch.zeros((batch_size, n_cat_features, 1)).bool()
      # [NOTE]: Everything is unmasked by default.
      if (
          data_.categorical_value_embeddings is None
          or data_.categorical_value_padding is None
      ):
        cat_vals_all = None
        cat_padding = None
      else:
        cat_vals_all = torch.repeat_interleave(
            data_.categorical_value_embeddings.unsqueeze(0),
            repeats=batch_size,
            dim=0,
        ).float()
        cat_padding = torch.repeat_interleave(
            data_.categorical_value_padding.unsqueeze(0),
            repeats=batch_size,
            dim=0,
        ).bool()

      if data_.encoded_categorical_ordinal is None:
        cat_y_vals = None
        cat_loss_weights = None
      else:
        cat_y_vals = data_.encoded_categorical_ordinal.unsqueeze(-1).long()
        cat_loss_weights = torch.ones_like(cat_y_vals).float()

      cat_inference_inputs = types.IngesTablesInferenceInputs(
          x_keys=cat_key_emb,
          x_vals=cat_val_emb,
          mask=cat_mask,  # Generated during training.
          missing=cat_missing,
          x_vals_all=cat_vals_all,
          padding=cat_padding,
      )
      cat_training_inputs = types.IngesTablesTrainingInputs(
          y_vals=cat_y_vals,
          loss_weights=cat_loss_weights,
      )

      # self.assert_cat_inference_shapes(
      #     cat_inference_inputs, n_cat_features, n_dims, batch_size, 8
      # )
      # self.assert_training_shapes(
      #     cat_training_inputs,
      #     n_cat_features,
      #     batch_size,
      # )
    ### Numeric features. ######################################################
    if data_.encoded_feature_names.numeric is None:
      n_num_features = 0
    else:
      n_num_features = len(data_.encoded_feature_names.numeric)

    if n_num_features == 0:
      num_inference_inputs = None
      num_training_inputs = None
    else:

      if data_.encoded_feature_names.numeric is None:
        num_key_emb = None
      else:
        num_key_emb = torch.repeat_interleave(
            data_.encoded_feature_names.numeric.unsqueeze(0),
            repeats=batch_size,
            dim=0,
        ).float()
      if data_.encoded_numeric is None:
        num_val_emb = None
      else:
        num_val_emb = data_.encoded_numeric.float()
      # TODO(joetoth): Get real missing.
      num_missing = torch.zeros((batch_size, n_num_features, 1)).bool()
      num_mask = torch.zeros((batch_size, n_num_features, 1)).bool()

      num_inference_inputs = types.IngesTablesInferenceInputs(
          x_keys=num_key_emb,
          x_vals=num_val_emb,
          mask=num_mask,  # Everything is unmasked by default.
          missing=num_missing,
      )

      num_y_vals = data_.raw_numeric[:batch_size].unsqueeze(-1).float()
      num_loss_weights = torch.ones_like(num_y_vals).float()
      num_training_inputs = types.IngesTablesTrainingInputs(
          y_vals=num_y_vals, loss_weights=num_loss_weights
      )

      # self.assert_num_inference_shapes(
      #     num_inference_inputs, n_num_features, n_dims, batch_size, n_bins
      # )
      # self.assert_training_shapes(
      #     num_training_inputs,
      #     n_num_features,
      #     batch_size,
      # )

    ### String features. #######################################################
    if data_.encoded_feature_names.string is None:
      n_str_features = 0
    else:
      n_str_features = len(data_.encoded_feature_names.string)

    if n_str_features == 0:
      str_inference_inputs = None
    else:
      if data_.encoded_feature_names.string is None:
        str_key_emb = None
      else:
        str_key_emb = torch.repeat_interleave(
            data_.encoded_feature_names.string.unsqueeze(0),
            repeats=batch_size,
            dim=0,
        ).float()
      if data_.encoded_string is None:
        str_val_emb = None
      else:
        str_val_emb = data_.encoded_string.float()

      # [NOTE]: String features are never masked.
      str_mask = torch.zeros((batch_size, n_str_features, 1)).bool()
      str_missing = torch.zeros((batch_size, n_str_features, 1)).bool()

      str_inference_inputs = types.IngesTablesInferenceInputs(
          x_keys=str_key_emb,
          x_vals=str_val_emb,
          mask=str_mask,
          missing=str_missing,
      )
      # self.assert_str_inference_shapes(
      #     str_inference_inputs, n_str_features, n_dims, batch_size
      # )

    inference_inputs, training_inputs = {}, {}
    if cat_inference_inputs is not None:
      inference_inputs["cat"] = cat_inference_inputs
    if num_inference_inputs is not None:
      inference_inputs["num"] = num_inference_inputs
    if str_inference_inputs is not None:
      inference_inputs["str"] = str_inference_inputs

    if cat_training_inputs is not None:
      training_inputs["cat"] = cat_training_inputs
    if num_training_inputs is not None:
      training_inputs["num"] = num_training_inputs

    return (inference_inputs, training_inputs)

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

  def _mask(
      self,
      task_info: types.TaskInfo,
      masking_strategy: masking.MaskingStrategy,
      inference_inputs: Dict[str, types.IngesTablesInferenceInputs],
      training_inputs: Dict[str, types.IngesTablesTrainingInputs],
  ) -> Tuple[
      Dict[str, types.IngesTablesInferenceInputs],
      Dict[str, types.IngesTablesTrainingInputs],
  ]:
    """Mask the inference inputs."""
    if masking_strategy is None:
      return inference_inputs, training_inputs

    target_head = "num" if task_info.task_type == "regression" else "cat"
    if task_info.task_type not in ["classification", "regression"]:
      raise NotImplementedError("Task type not supported.")

    for key in inference_inputs.keys():
      target_available = key == target_head
      mask, loss_weights = masking_strategy.generate_mask(
          target_available=target_available,
          inference_inputs=inference_inputs[key],
          feature_type=key,
      )
      inference_inputs[key].mask = mask
      if key in training_inputs:
        training_inputs[key].loss_weights = loss_weights

      if target_available:
        logging.info(
            "Generated target-aware mask since task_type is `%s` and"
            " inference_inputs[key] is `%s`",
            task_info.task_type,
            key,
        )

    return inference_inputs, training_inputs

  def _train_step(
      self, dataset_key: str, batch: encoders.PreprocessedInputs
  ) -> None:
    """Run a single training step."""
    # Updates the model weights, training step count, and the training dataset's
    # iterator.
    # Following AMP recipe here:
    # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
    self._model.train()
    # dataset_key = list(self._pipeline.dataset_keys)[0]

    task_info = self._pipeline.get_task_info(dataset_key)
    logging.info(
        "Training step %s on dataset: %s", self._train_step_count, dataset_key
    )
    inference_inputs, training_inputs = self._format_inputs(batch)

    # Mask training and inference inputs
    inference_inputs, training_inputs = self._mask(
        task_info,
        self._train_masking_strategy,
        inference_inputs=inference_inputs,
        training_inputs=training_inputs,
    )

    inference_inputs, training_inputs = utils.move_to_device(
        inference_inputs=inference_inputs,
        training_inputs=training_inputs,
        device=self._device,
    )

    self._optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=self._device.type,
        # amp_dtype=None will behave reasonably in most case, but T5 can be
        # unstable with float16, so we allow overriding this explicitly.
        dtype=self._amp_dtype,
        enabled=self._enable_amp,
    ):
      logits = self._model(inference_inputs)

    losses_dict = self._model_module.loss(logits, training_inputs)

    if not losses_dict:
      return

    loss = torch.sum(torch.stack(list(losses_dict.values())))
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
      dataloaders: Dict[str, torch.utils.data.DataLoader],
  ) -> None:
    """Evaluates the model on test datasets."""
    if self._global_rank != 0:
      return
    self._model.eval()

    for dataset_key, eval_dataloader in dataloaders.items():
      task_info = self._pipeline.get_task_info(dataset_key)
      target_head = "num" if task_info.task_type == "regression" else "cat"
      logging.info(
          "Evaluating on `%s %s` dataset on the `%s` task with the `%s` head",
          dataset_type,
          dataset_key,
          task_info.task_type,
          target_head,
      )
      batched_logits_dict = {}
      batched_training_inputs = {}
      for batch in eval_dataloader:
        inference_inputs, training_inputs = self._format_inputs(batch)

        # Mask training and inference inputs
        inference_inputs, training_inputs = self._mask(
            task_info,
            self._eval_masking_strategy,
            inference_inputs=inference_inputs,
            training_inputs=training_inputs,
        )

        inference_inputs, training_inputs = utils.move_to_device(
            inference_inputs=inference_inputs,
            training_inputs=training_inputs,
            device=self._device,
        )

        with torch.no_grad():
          with torch.autocast(
              device_type=self._device.type,
              dtype=self._amp_dtype,
              enabled=self._enable_amp,
          ):
            logits_dict = self._model(inference_inputs)

        # Convert to float as logits may be of type torch.bfloat16.

        for head_key in logits_dict.keys():
          if head_key not in batched_logits_dict:
            batched_logits_dict[head_key] = []
          batched_logits_dict[head_key].append(logits_dict[head_key].float())

        for head_key in logits_dict.keys():
          if head_key not in batched_training_inputs:
            batched_training_inputs[head_key] = []
          batched_training_inputs[head_key].append(training_inputs[head_key])

      # Collate all the batches into a single tensor.
      batched_logits_dict = {
          head_key: torch.cat(logits_dict, dim=0)
          for head_key, logits_dict in batched_logits_dict.items()
      }
      batched_training_inputs = {
          head_key: utils.training_inputs_collate_fn(training_inputs)
          for head_key, training_inputs in batched_training_inputs.items()
      }

      metrics_ = self._model_module.heads[target_head].compute_metrics(
          logits=batched_logits_dict[target_head],
          training_inputs=batched_training_inputs[target_head],
      )

      self._metrics_writer.write_model_metrics(
          self._train_step_count,
          dataset_type,
          dataset_key,
          metrics_,
      )
      logging.info(
          "Dataset: %s %s, Target head: %s, Metrics: %s",
          dataset_type,
          dataset_key,
          target_head,
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
    train_iter_idx = training_step % self._num_datasets
    train_iter = self._train_iters[train_iter_idx]
    dataset_key = self._dataset_key_list[train_iter_idx]

    return dataset_key, next(train_iter)

  def run(self) -> None:
    """Train, evaluate, and save checkpoints."""
    for step in range(self._num_train_steps):
      self._train_step_count = step
      is_last_step = step >= self._num_train_steps - 1

      dataset_key, batch = self._get_training_batch(step)
      self._train_step(dataset_key, batch)
      self._lr_scheduler.step()

      logging.info("Learning rate: %s", self._lr_scheduler.get_last_lr())
      self._metrics_writer.write_metric(
          self._train_step_count,
          (dataset_key,),
          "learning_rate",
          self._lr_scheduler.get_last_lr()[0],
      )

      if step % self._eval_every_steps == 0 or is_last_step:
        self._evaluate("test", self._test_dataloaders)
        self._evaluate("val", self._val_dataloaders)
      if step % self._checkpoint_every_steps == 0 or is_last_step:
        self._save_checkpoint()

    logging.info("Job finished")

  @property
  def eval_metrics(self) -> dict[str, dict[str, Any]]:
    if self._all_metrics is None:
      raise ValueError("No eval metrics. Make sure to call .run() first.")
    return self._all_metrics

  # def assert_cat_inference_shapes(
  #     self,
  #     inference_inputs: types.IngesTablesInferenceInputs,
  #     n_features: int,
  #     n_dims: int,
  #     batch_size: int,
  #     max_num_categories: int,
  # ):
  #   """Asserts the shapes of the data."""
  #   # Code to evaluate the dimensionality
  #   if not _DEBUG_ASSERT_INPUT_SHAPES.value:
  #     return
  #   utils.assert_equals(
  #       inference_inputs.x_keys.shape, (batch_size, n_features, n_dims)
  #   )
  #   utils.assert_equals(
  #       inference_inputs.x_vals.shape, (batch_size, n_features, n_dims)
  #   )
  #   # utils.assert_equals(
  #   #     inference_inputs.mask.shape, (batch_size, n_features, 1)
  #   # )
  #   if inference_inputs.missing is not None:
  #     utils.assert_equals(
  #         inference_inputs.missing.shape, (batch_size, n_features, 1)
  #     )
  #   assert inference_inputs.x_vals_all is not None
  #   utils.assert_equals(
  #       inference_inputs.x_vals_all.shape,
  #       (
  #           batch_size,
  #           n_features,
  #           max_num_categories,
  #           n_dims,
  #       ),
  #   )
  #   assert inference_inputs.padding is not None
  #   utils.assert_equals(
  #       inference_inputs.padding.shape,
  #       (batch_size, n_features, max_num_categories),
  #   )

  # def assert_num_inference_shapes(
  #     self,
  #     inference_inputs: types.IngesTablesInferenceInputs,
  #     n_features: int,
  #     n_dims: int,
  #     batch_size: int,
  #     n_bins: int,
  # ):
  #   """Asserts the shapes of the data."""
  #   # Code to evaluate the dimensionality
  #   if not _DEBUG_ASSERT_INPUT_SHAPES.value:
  #     return
  #   utils.assert_equals(
  #       inference_inputs.x_keys.shape, (batch_size, n_features, n_dims)
  #   )
  #   utils.assert_equals(
  #       inference_inputs.x_vals.shape, (batch_size, n_features, n_bins)
  #   )
  #   # utils.assert_equals(
  #   #     inference_inputs.mask.shape, (batch_size, n_features, 1)
  #   # )
  #   if inference_inputs.missing is not None:
  #     utils.assert_equals(
  #         inference_inputs.missing.shape, (batch_size, n_features, 1)
  #     )

  # def assert_str_inference_shapes(
  #     self,
  #     inference_inputs: types.IngesTablesInferenceInputs,
  #     n_features: int,
  #     n_dims: int,
  #     batch_size: int,
  # ):
  #   """Asserts the shapes of the data."""
  #   # Code to evaluate the dimensionality
  #   if not _DEBUG_ASSERT_INPUT_SHAPES.value:
  #     return
  #   utils.assert_equals(
  #       inference_inputs.x_keys.shape, (batch_size, n_features, n_dims)
  #   )
  #   utils.assert_equals(
  #       inference_inputs.x_vals.shape, (batch_size, n_features, n_dims)
  #   )
  #   # utils.assert_equals(
  #   #     inference_inputs.mask.shape, (batch_size, n_features, 1)
  #   # )
  #   if inference_inputs.missing is not None:
  #     utils.assert_equals(
  #         inference_inputs.missing.shape, (batch_size, n_features, 1)
  #     )

  # def assert_training_shapes(
  #     self,
  #     training_inputs: types.IngesTablesTrainingInputs,
  #     n_features: int,
  #     batch_size: int,
  # ):
  #   """Asserts the shapes of the data."""
  #   if not _DEBUG_ASSERT_INPUT_SHAPES.value:
  #     return
  #   utils.assert_equals(
  #       training_inputs.y_vals.shape, (batch_size, n_features, 1)
  #   )
  #   utils.assert_equals(
  #       training_inputs.loss_weights.shape, (batch_size, n_features, 1)
  #   )

  # def assert_training_shapes(
  #     self,
  #     training_inputs: types.IngesTablesTrainingInputs,
  #     n_features: int,
  #     batch_size: int,
  # ):
  #   """Asserts the shapes of the data."""
  #   if not _DEBUG_ASSERT_INPUT_SHAPES.value:
  #     return
  #   utils.assert_equals(
  #       training_inputs.y_vals.shape, (batch_size, n_features, 1)
  #   )
  #   utils.assert_equals(
  #       training_inputs.loss_weights.shape, (batch_size, n_features, 1)
  #   )
