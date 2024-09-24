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

from typing import Any

from absl import logging
from etils import epath
from etils import etree
from ingestables.torch import utils
from ingestables.torch.model import ingestables
import torch
from torch import optim
import torch.distributed as dist  # pylint:disable=g-importing-member
from torch.nn.parallel import DistributedDataParallel as DDP  # pylint:disable=g-importing-member
from torch.utils import data


_LARGE_INT = int(1e100)


def train(
    model: ingestables.Model,
    device: torch.device,
    train_loaders: dict[str, data.DataLoader],
    optimizer: optim.Optimizer,
    num_steps: int,
    log_loss_every_steps: int,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
  """Train."""
  # Following AMP recipe here:
  # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
  scaler = torch.amp.GradScaler()
  model.train()
  train_losses = {ds_key: 0.0 for ds_key in train_loaders.keys()}
  ds_key_list = list(train_loaders.keys())
  num_ds = len(ds_key_list)
  train_iters = [iter(train_loader) for train_loader in train_loaders.values()]
  model_module = model.module if isinstance(model, DDP) else model
  for step in range(num_steps):
    ds_ix = step % num_ds
    ds_key = ds_key_list[ds_ix]  # type: str
    train_iter = train_iters[ds_ix]
    inference_inputs, training_inputs, _ = next(train_iter)
    inference_inputs, training_inputs = etree.map(
        lambda t: t.to(device),
        (inference_inputs, training_inputs),
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        # amp_dtype=None will behave reasonably in most case, but T5 can be
        # unstable with float16, so we allow overriding this explicitly.
        dtype=amp_dtype,
    ):
      logits = model(inference_inputs)
      losses_dict = model_module.loss(logits, training_inputs)
    if not losses_dict:
      continue
    loss = torch.sum(torch.stack(list(losses_dict.values())))
    scaler.scale(loss).backward()
    # TODO(scottyak): Add grad norm clipping.
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
    scaler.step(optimizer)
    scaler.update()

    loss_item = loss.item()
    if step % log_loss_every_steps == 0:
      logging.info(
          "[%s/%s] Loss: %s",
          step,
          num_steps,
          loss_item,
      )
    train_losses[ds_key] += loss_item
  return train_losses


def store_metrics_kwargs(
    metrics_dict: dict[str, Any],
    logits_dict: dict[str, torch.Tensor],
    training_inputs: dict[str, dict[str, torch.Tensor]],
    eval_inputs: dict[str, dict[str, torch.Tensor]],
) -> None:
  """Store in metrics_dict."""
  for head_key, logits in logits_dict.items():
    if head_key not in eval_inputs.keys():
      continue
    if head_key not in metrics_dict.keys():
      metrics_dict[head_key] = {}
    if "logits" not in metrics_dict[head_key].keys():
      metrics_dict[head_key]["logits"] = []
    metrics_dict[head_key]["logits"].append(logits)
    if "training_inputs" not in metrics_dict[head_key].keys():
      metrics_dict[head_key]["training_inputs"] = {}
    for key, training_input in training_inputs[head_key].items():
      if key not in metrics_dict[head_key]["training_inputs"].keys():
        metrics_dict[head_key]["training_inputs"][key] = []
      metrics_dict[head_key]["training_inputs"][key].append(training_input)
    if "eval_inputs" not in metrics_dict[head_key].keys():
      metrics_dict[head_key]["eval_inputs"] = {}
    for key, eval_input in eval_inputs[head_key].items():
      if key not in metrics_dict[head_key]["eval_inputs"].keys():
        metrics_dict[head_key]["eval_inputs"][key] = []
      metrics_dict[head_key]["eval_inputs"][key].append(eval_input)


def concatenate_metrics_kwargs(
    metrics_dict: dict[str, Any],
) -> dict[str, Any]:
  return etree.map(
      torch.concat,
      metrics_dict,
      is_leaf=lambda x: isinstance(x, list),
  )


def test(
    model: ingestables.Model,
    device: torch.device,
    test_loaders: dict[str, data.DataLoader],
) -> dict[str, dict[str, Any]]:
  """Test."""
  model.eval()
  all_ds_metrics = {}
  model_module = model.module if isinstance(model, DDP) else model
  for test_ds_key, test_loader in test_loaders.items():
    if test_ds_key not in all_ds_metrics.keys():
      all_ds_metrics[test_ds_key] = {}
    metrics_dict = {}
    with torch.no_grad():
      for inference_inputs, training_inputs, eval_inputs in test_loader:
        inference_inputs, training_inputs, eval_inputs = etree.map(
            lambda t: t.to(device),
            (inference_inputs, training_inputs, eval_inputs),
        )
        logits_dict = model(inference_inputs)
        store_metrics_kwargs(
            metrics_dict, logits_dict, training_inputs, eval_inputs
        )
      metrics_kwargs_dict = concatenate_metrics_kwargs(metrics_dict)
      for head_key, metrics_kwargs in metrics_kwargs_dict.items():
        metrics = model_module.heads[head_key].metrics(
            logits=metrics_kwargs["logits"],
            training_inputs=metrics_kwargs["training_inputs"],
            eval_inputs=metrics_kwargs["eval_inputs"],
        )
        all_ds_metrics[test_ds_key][head_key] = metrics
  return all_ds_metrics


def save_model(model, optimizer, output_path: epath.Path):
  """Save model."""
  # torch.save() requires existing output_dir.
  output_path.mkdir(parents=True, exist_ok=True)
  with (output_path / "model.pt").open("wb") as f:
    torch.save(model.state_dict(), f)
  with (output_path / "checkpoint.tar").open("wb") as f:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        f,
    )


def train_and_test(
    model: ingestables.Model,
    train_datasets: dict[str, data.Dataset],
    test_datasets: dict[str, data.Dataset],
    train_steps: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    momentum: float,
    log_loss_every_steps: int,
    workdir: str,
    local_rank: int = 0,
    nproc_per_node: int = 1,
    node_rank: int = 0,
    world_size: int = 1,
    num_data_workers: int = 1,
    seed: int = 1,
):
  """Performs both training and testing of the model on the given datasets."""
  logging.info(
      "local_rank=%s, nproc_per_node=%s, node_rank=%s, world_size=%s, seed=%s",
      local_rank,
      nproc_per_node,
      node_rank,
      world_size,
      seed,
  )

  gpu_ok = False  # whether the GPU can benefit from compilation.
  if torch.cuda.device_count() > 0:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    device_cap = torch.cuda.get_device_capability()
    # Only models that run on V100, A100, or H100 benefit from compilation.
    if device_cap in ((7, 0), (8, 0), (9, 0)):
      gpu_ok = True
  else:
    device = torch.device("cpu")
  logging.info("Using device: %s", device)

  utils.seed_everything(seed)

  if world_size > 1:
    global_rank = node_rank * nproc_per_node + local_rank
    dist.init_process_group(
        backend="NCCL",
        rank=global_rank,
        world_size=world_size,
    )
  else:
    global_rank = 0

  model = model.to(device)
  if gpu_ok:
    model.compile(dynamic=False)
  if world_size > 1:
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

  # TODO(mononito): Add Adam, AdamW, Adagrad optimizers
  optimizer = optim.SGD(
      model.parameters(),
      lr=learning_rate,
      momentum=momentum,
  )
  save_path = epath.Path(workdir) / f"lr={learning_rate},mom={momentum}"

  train_loaders = {
      train_dataset_key: data.DataLoader(
          train_dataset,
          batch_size=train_batch_size,
          sampler=data.RandomSampler(
              train_dataset,  # pytype: disable=wrong-arg-types
              replacement=True,
              num_samples=_LARGE_INT,
              generator=torch.Generator().manual_seed(seed),
          ),
          num_workers=num_data_workers,
      )
      for train_dataset_key, train_dataset in train_datasets.items()
  }
  if global_rank == 0:
    test_loaders = {
        test_dataset_key: data.DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
        )
        for test_dataset_key, test_dataset in test_datasets.items()
    }

  logging.info("TRAIN START")
  train_loss = train(
      model,
      device=device,
      train_loaders=train_loaders,
      optimizer=optimizer,
      num_steps=train_steps,
      log_loss_every_steps=log_loss_every_steps,
      # We can make this configurable after the training loop is refactored to
      # to take in a configuration object.
      amp_dtype=None,
  )
  logging.info("TRAIN END")
  logging.info("train_loss: %s", train_loss)

  if global_rank == 0:
    metrics = test(model, device, test_loaders)  # pylint: disable=undefined-variable
    logging.info("Test metrics: %s", metrics)

  if workdir:
    if global_rank == 0:
      logging.info("Saving model on rank 0 to: %s", save_path)
      save_model(model, optimizer, save_path)
    if world_size > 1:
      dist.barrier()

  logging.info("Job finished")
