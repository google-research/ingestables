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

"""IngesTables built-in heads."""

from ingestables.torch import types
from ingestables.torch.train import metrics
import numpy as np
from sklearn import metrics as sklearn_metrics
import torch
from torch import nn
from torch.nn import functional as F


# Inputs have shape:
#   logits: [batch, num_cat_feats, num_classes]
#   labels: [batch, num_cat_feats]
# Want to parallelize over all categorical features.
_cross_entropy_fn = torch.vmap(F.cross_entropy, in_dims=1, out_dims=1)


class IngesTablesClassification(nn.Module):
  """Converts the z_emb to logits."""

  def __init__(
      self,
      aligner: nn.Module,
      kv_combiner: nn.Module,
      max_num_classes: int,
  ):
    """Constructor.

    Args:
      aligner: See `CategoricalAligner` in aligner.py for more info.
      kv_combiner: See kv_combiner.py for more info.
      max_num_classes: Max number of classes among all categorical features.
    """
    super().__init__()
    self.aligner = aligner
    self.kv_combiner = kv_combiner
    self.max_num_classes = max_num_classes

  def __repr__(self):
    return (
        f"IngesTablesClassification(aligner={self.aligner},"
        + f" kv_combiner={self.kv_combiner},"
        + f" max_num_classes={self.max_num_classes})"
    )

  def forward(
      self,
      z_emb: torch.Tensor,
      inference_inputs: types.IngesTablesInferenceInputs,
  ) -> torch.Tensor:
    """Convert embedding to logits.

    x_keys and x_vals_all are fed into the aligner to construct embeddings for
    all possible categories. Then we take a dot product between z_emb and these
    embeddings to obtain the logits to be fed into softmax. Since not all
    categorical features have the same number of categories, padding is applied
    to ensure that argmax will never choose the padding category as the
    predicted class.

    Args:
      z_emb: [..., num_cat_feats, z_dim] float tensor. The output of the last
        encoder layer.
      inference_inputs: IngesTablesInferenceInputs dataclass.

    Returns:
      [..., num_cat_feats, max_num_classes] float tensor.
        Logits for classification wherever padding is 1.
    """
    x_keys = inference_inputs.x_keys
    # x_keys.shape: [batch, num_cat_feats, x_key_dim]
    x_vals_all = inference_inputs.x_vals_all
    # x_vals_all.shape: [batch, num_cat_feats, max_num_classes, x_val_dim]
    padding = inference_inputs.padding
    # Expand x_key_embs [..., x_key_dim] -> [..., max_num_classes, x_key_dim].
    expected_shape = x_vals_all.shape[:-1] + (-1,)  # pylint: disable=attribute-error
    # expected_shape = [batch_size, num_cat_feats, max_num_classes, -1]
    x_keys = x_keys.unsqueeze(-2)  # pylint: disable=attribute-error
    # x_keys.shape: [batch, num_cat_feats, 1, x_key_dim]
    x_keys = x_keys.expand(*expected_shape)
    # x_keys.shape: [batch, num_cat_feats, x_key_dim]
    # mask.shape = missing.shape = [..., num_classes, 1]
    z_key_embs_all, z_val_embs_all = self.aligner(x_keys, x_vals_all)
    z_embs_all = self.kv_combiner(z_key_embs_all, z_val_embs_all)
    # z_embs_all.shape = [..., num_cat_feats, max_num_classes, z_dim]
    # z_embs.shape = [..., num_cat_feats, z_dim]
    logits = torch.einsum("...nd,...nkd->...nk", z_emb, z_embs_all)
    # logits.shape = [..., num_cat_feats, max_num_classes]
    logits = torch.where(padding, logits, float("-inf"))
    return logits

  def loss(
      self,
      logits: torch.Tensor,
      training_inputs: types.IngesTablesTrainingInputs,
  ) -> torch.Tensor:
    """Compute the cross entropy loss.

    Args:
      logits: [..., num_cat_feats, max_num_classes] float tensor. The output of
        forward pass.
      training_inputs: IngesTablesTrainingInputs dataclass.

    Returns:
      float scalar containing the loss of this batch, weighted mean.
    """
    y_vals = training_inputs.y_vals  # [batch, num_cat_feats, 1]
    y_vals = y_vals.squeeze(-1)  # [batch, num_cat_feats]  # pylint: disable=attribute-error
    loss_weights = training_inputs.loss_weights  # [batch, num_cat_feats, 1]
    loss_weights = loss_weights.squeeze()  # [batch, num_cat_feats]  # pylint: disable=attribute-error
    loss = _cross_entropy_fn(
        logits,  # [batch, num_cat_feats, num_classes]
        y_vals,  # [batch, num_cat_feats]
        reduction="none",
    )  # [batch, num_cat_feats]
    return torch.mean(loss * loss_weights)  # scalar

  # TODO(joetoth): Merge with sklearn_model.py
  def compute_metrics(
      self,
      logits: torch.Tensor,
      training_inputs: types.IngesTablesTrainingInputs,
  ) -> types.ClassificationMetrics:
    """Compute classifcation metrics.

    The taget feature is always the first one.

    Args:
      logits: [..., num_cat_feats, max_num_classes] float tensor. The output of
        forward pass.
      training_inputs: IngesTablesTrainingInputs dataclass.

    Returns:
      ClassificationMetrics.
    """
    y_vals = training_inputs.y_vals
    # y_vals.shape: [batch, num_cat_feats, 1]
    # target_index = eval_inputs.target_index
    # target_index.shape: [batch, 1, 1]
    # logits = torch.take_along_dim(logits, target_index, dim=-2).squeeze(
    #     1
    # )  # [batch, max_num_classes]
    # y_true = torch.take_along_dim(y_vals, target_index, dim=-2).squeeze(
    #     1
    # )  # [batch, 1]
    # The target is always the first feature.
    if torch.isinf(logits).any():
      # Then max_num_categories > n_classes
      n_classes = torch.argmin(logits[0, 0]).item()
    else:
      n_classes = len(logits[0, 0])
    logits = logits[..., :1, :].squeeze(1)  # [batch, max_num_classes]
    y_true = y_vals[..., :1, :].squeeze(1)  # [batch, 1]
    y_pred = torch.argmax(logits, dim=-1)
    y_probs = torch.softmax(logits, dim=-1)
    # Copy to numpy
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_probs = y_probs.detach().cpu().numpy()
    log_loss = sklearn_metrics.log_loss(
        y_true, y_probs[:, :n_classes], labels=np.arange(n_classes)
    )
    accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)

    # TODO(mononito): Make this configurable.
    f1_score = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
    auc_roc = sklearn_metrics.roc_auc_score(
        y_true, y_probs[:, 1], multi_class="ovr", average="macro"
    )
    if y_probs.shape[-1] == 2:
      auc_pr = sklearn_metrics.average_precision_score(y_true, y_probs[:, 1])
    else:
      auc_pr = None

    return types.ClassificationMetrics(
        accuracy=accuracy,
        log_loss=log_loss,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        f1_score=f1_score,
    )


class IngesTablesRegression(nn.Module):
  """Converts the z_emb to normalized numeric prediction."""

  def __init__(self, z_dim: int):
    """Constructor."""
    super().__init__()
    self.linear = nn.Linear(in_features=z_dim, out_features=1, bias=True)

  def __repr__(self):
    return f"IngesTablesRegression(linear={self.linear})"

  def forward(
      self,
      z_emb: torch.Tensor,
      inference_inputs: types.IngesTablesInferenceInputs,
  ) -> torch.Tensor:
    """Convert embedding to normalized numeric prediction.

    This is done through a linear transformation.

    Args:
      z_emb: [..., z_dim] float tensor. The output of the last encoder layer.
      inference_inputs: Unused.

    Returns:
      [..., 1] float tensor. Output for regression.
    """
    del inference_inputs
    return self.linear(F.relu(z_emb))

  def loss(
      self,
      logits: torch.Tensor,
      training_inputs: types.IngesTablesTrainingInputs,
  ) -> torch.Tensor:
    """Compute the mean squared error.

    Args:
      logits: [..., num_num_feats, 1] float tensor. The output of forward pass.
      training_inputs: IngesTablesTrainingInputs dataclass.

    Returns:
      float scalar containing the loss of this batch, weighted mean.
    """
    y_vals = training_inputs.y_vals
    loss_weights = training_inputs.loss_weights
    loss = torch.square(logits - y_vals)  # [batch, num_num_feats, 1]
    return torch.mean(loss * loss_weights)  # scalar

  def compute_metrics(
      self,
      logits: torch.Tensor,
      training_inputs: types.IngesTablesTrainingInputs,
  ) -> types.RegressionMetrics:
    """Compute regression metrics.

    The taget feature is always the first one.

    Args:
      logits: [..., num_num_feats, 1] float tensor. The output of forward pass.
      training_inputs: IngesTablesTrainingInputs dataclass.

    Returns:
      RegressionMetrics.
    """
    # y_vals.shape: [batch, num_num_feats, 1]
    # target_index = eval_inputs.target_index
    # target_index.shape: [batch, 1, 1]
    # logits = torch.take_along_dim(logits, target_index, dim=-2)  # [batch, 1]
    # y_vals = torch.take_along_dim(y_vals, target_index, dim=-2)  # [batch, 1]
    # The target is always the first feature.
    y_true = training_inputs.y_vals[..., 0, :].detach().cpu().numpy()
    y_pred = logits[..., 0, :].detach().cpu().numpy()
    # y_true and y_pred are of shape [batch,]

    return types.RegressionMetrics(
        mean_squared_error=sklearn_metrics.mean_squared_error(y_true, y_pred),
        mean_absolute_error=sklearn_metrics.mean_absolute_error(y_true, y_pred),
        root_mean_squared_error=np.sqrt(
            sklearn_metrics.mean_squared_error(y_true, y_pred)
        ),
        r_squared=sklearn_metrics.r2_score(y_true, y_pred),
    )


class SklearnRegression(metrics.ComputeMetrics):
  """Not really a head, but a way to compute regression sklearn_metrics."""

  def __repr__(self):
    return "SklearnRegression()"

  def compute_metrics(
      self,
      *,
      y_true: np.ndarray,
      y_probs: np.ndarray,
  ) -> types.Metrics:
    return types.RegressionMetrics(
        mean_squared_error=sklearn_metrics.mean_squared_error(y_true, y_probs),
        mean_absolute_error=sklearn_metrics.mean_absolute_error(
            y_true, y_probs
        ),
        root_mean_squared_error=np.sqrt(
            sklearn_metrics.mean_squared_error(y_true, y_probs)
        ),
        r_squared=sklearn_metrics.r2_score(y_true, y_probs),
    )

  def loss(
      self,
      *,
      y_true: np.ndarray,
      y_probs: np.ndarray,
  ) -> float:
    return sklearn_metrics.mean_squared_error(y_true, y_probs)


class SklearnClassification(metrics.ComputeMetrics):
  """Not really a head, but a way to compute classification metrics."""

  def __repr__(self):
    return "SklearnClassification()"

  def compute_metrics(
      self,
      *,
      y_true: np.ndarray,
      y_probs: np.ndarray,
  ) -> types.Metrics:
    y_pred = np.argmax(y_probs, axis=1)
    accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
    # TODO(mononito): Make this configurable.
    f1_score = sklearn_metrics.f1_score(y_true, y_pred, average="macro")

    # Binary probability estimates correspond to the probability of the class
    # with the greater label i.e. `estimator.classes_[1]`
    if y_probs.shape[-1] == 2:
      y_probs = y_probs[:, 1]

    auc_roc = sklearn_metrics.roc_auc_score(
        y_true, y_probs, multi_class="ovr", average="macro"
    )
    if y_probs.shape[-1] == 2:
      auc_pr = sklearn_metrics.average_precision_score(y_true, y_probs)
    else:
      auc_pr = None
    log_loss = sklearn_metrics.log_loss(y_true, y_probs)

    return types.ClassificationMetrics(
        accuracy=accuracy,
        log_loss=log_loss,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        f1_score=f1_score,
    )
