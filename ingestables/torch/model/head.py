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

import dataclasses

from sklearn import metrics
import torch
from torch import nn
from torch.nn import functional as F


# Inputs have shape:
#   logits: [batch, num_cat_feats, num_classes]
#   labels: [batch, num_cat_feats]
# Want to parallelize over all categorical features.
_cross_entropy_fn = torch.vmap(F.cross_entropy, in_dims=1, out_dims=1)


@dataclasses.dataclass
class ClassificationConfig:
  # Max number of classes in all the categorical features.
  # For example, if you have 3 categorical features with [2, 6, 10] classes,
  # and you want to reconstruct all of them as part of training, you should
  # set "max_num_classes" to 10.
  max_num_classes: int


@dataclasses.dataclass
class ClassificationMetrics:
  accuracy: float
  loss: float
  auc_roc: float | None = None
  auc_pr: float | None = None


class Classification(nn.Module):
  """Converts the z_emb to logits."""

  def __init__(
      self,
      config: ClassificationConfig,
      *,
      aligner: nn.Module,
      kv_combiner: nn.Module,
      **kwargs,
  ):
    """Constructor.

    Args:
      config: ClassificationConfig.
      aligner: See `CatAligner` in aligner.py for more info.
      kv_combiner: See kv_combiner.py for more info.
      **kwargs: Unused.
    """
    super().__init__()
    self.aligner = aligner
    self.kv_combiner = kv_combiner
    self.max_num_classes = config.max_num_classes

  def forward(
      self,
      z_emb: torch.Tensor,
      inference_inputs: dict[str, torch.Tensor],
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
      inference_inputs: dict containing:
        "x_keys": [..., num_cat_feats, x_key_dim] float tensor.
          This is the feature key of the categorical feature to be treated as
          the label.
        "x_vals_all": [..., num_cat_feats, max_num_classes, x_val_dim] float
          tensor. Corresponds to all possible categories for that categorical
          feature, but padded up the max_num_classes.
        "padding": [..., num_cat_feats, max_num_classes] bool tensor.
          0 wherever x_vals is just padding, 1 otherwise. This forces softmax
          to predict 0 probability for classes that don't actually exist.

    Returns:
      [..., num_cat_feats, max_num_classes] float tensor.
        Logits for classification wherever padding is 1.
    """  # fmt: skip
    x_keys = inference_inputs["x_keys"]
    # x_keys.shape: [batch, num_cat_feats, x_key_dim]
    x_vals_all = inference_inputs["x_vals_all"]
    # x_vals_all.shape: [batch, num_cat_feats, max_num_classes, x_val_dim]
    padding = inference_inputs["padding"]
    # Expand x_key_embs [..., x_key_dim] -> [..., max_num_classes, x_key_dim].
    expected_shape = x_vals_all.shape[:-1] + (-1,)
    # expected_shape = [batch_size, num_cat_feats, max_num_classes, -1]
    x_keys = x_keys.unsqueeze(-2)
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
      training_inputs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Compute the cross entropy loss.

    Args:
      logits: [..., num_cat_feats, max_num_classes] float tensor. The output of
        forward pass.
      training_inputs: dict containing:
        "y_vals": [..., num_cat_feats, 1] int64 tensor.
          This is the index of the category.
        "loss_weights": [..., num_cat_feats, 1] float tensor.
          This is multiplied element-wise to the loss for each instance and for
          each feature.

    Returns:
      float scalar containing the loss of this batch, weighted mean.
    """  # fmt: skip
    y_vals = training_inputs["y_vals"]  # [batch, num_cat_feats, 1]
    y_vals = y_vals.squeeze(2)  # [batch, num_cat_feats]
    loss_weights = training_inputs["loss_weights"]  # [batch, num_cat_feats, 1]
    loss_weights = loss_weights.squeeze(2)  # [batch, num_cat_feats]
    loss = _cross_entropy_fn(
        logits,  # [batch, num_cat_feats, num_classes]
        y_vals,  # [batch, num_cat_feats]
        reduction="none",
    )  # [batch, num_cat_feats]
    return torch.mean(loss * loss_weights)  # scalar

  def metrics(
      self,
      logits: torch.Tensor,
      training_inputs: dict[str, torch.Tensor],
      eval_inputs: dict[str, torch.Tensor],
  ) -> ClassificationMetrics:
    """Compute classifcation metrics at the target index.

    Args:
      logits: [..., num_cat_feats, max_num_classes] float tensor. The output of
        forward pass.
      training_inputs: dict containing:
        "y_vals": [..., num_cat_feats, 1] int64 tensor.
          This is the index of the category.
      eval_inputs: dict containing:
        "target_index": [..., 1, 1] int64 tensor.
          This is the index along the features axis whose eval metrics we care
          about.

    Returns:
      ClassificationMetrics.
    """  # fmt: skip
    y_vals = training_inputs["y_vals"]
    # y_vals.shape: [batch, num_cat_feats, 1]
    target_index = eval_inputs["target_index"]
    # target_index.shape: [batch, 1, 1]
    logits = torch.take_along_dim(logits, target_index, dim=-2).squeeze(
        1
    )  # [batch, max_num_classes]
    y_true = torch.take_along_dim(y_vals, target_index, dim=-2).squeeze(
        1
    )  # [batch, 1]
    y_pred = torch.argmax(logits, dim=-1)
    y_probs = torch.softmax(logits, dim=-1)
    # Copy to numpy
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    y_probs = y_probs.detach().cpu().numpy()
    loss = metrics.log_loss(y_true, y_probs)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    if y_probs.shape[-1] == 2:
      auc_roc = metrics.roc_auc_score(y_true, y_probs[:, 1])
      auc_pr = metrics.average_precision_score(y_true, y_probs[:, 1])
    else:
      auc_roc = None
      auc_pr = None
    return ClassificationMetrics(
        accuracy=accuracy,
        loss=loss,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
    )


@dataclasses.dataclass
class RegressionConfig:
  # The dimension of the last encoder layer.
  z_dim: int


@dataclasses.dataclass
class RegressionMetrics:
  # The dimension of the last encoder layer.
  mse: float


class Regression(nn.Module):
  """Converts the z_emb to normalized numeric prediction."""

  def __init__(self, config: RegressionConfig, **kwargs):
    """Constructor."""
    super().__init__()
    self.linear = nn.Linear(in_features=config.z_dim, out_features=1, bias=True)

  def forward(
      self,
      z_emb: torch.Tensor,
      inference_inputs: dict[str, torch.Tensor],
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
    return F.relu(self.linear(z_emb))

  def loss(
      self,
      logits: torch.Tensor,
      training_inputs: dict[str, torch.Tensor],
  ) -> torch.Tensor:
    """Compute the mean squared error.

    Args:
      logits: [..., num_num_feats, 1] float tensor. The output of forward pass.
      training_inputs: dict containing:
        "y_vals": [..., num_cat_feats, 1] int64 tensor.
          This is the index of the category.
        "loss_weights": [..., num_cat_feats, 1] float tensor.
          This is multiplied element-wise to the loss for each instance and for
          each categorical feature.

    Returns:
      float scalar containing the loss of this batch, weighted mean.
    """  # fmt: skip
    y_vals = training_inputs["y_vals"]
    loss_weights = training_inputs["loss_weights"]
    loss = torch.square(logits - y_vals)  # [batch, num_num_feats, 1]
    return torch.mean(loss * loss_weights)  # scalar

  def metrics(
      self,
      logits: torch.Tensor,
      training_inputs: dict[str, torch.Tensor],
      eval_inputs: dict[str, torch.Tensor],
  ) -> RegressionMetrics:
    """Compute regression metrics at the target index.

    Args:
      logits: [..., num_num_feats, 1] float tensor. The output of forward pass.
      training_inputs: dict containing:
        "y_vals": [..., num_cat_feats, 1] int64 tensor.
          This is the index of the category.
      eval_inputs: dict containing:
        "target_index": [..., 1, 1] int64 tensor.
          This is the index along the features axis whose eval metrics we care
          about.

    Returns:
      RegressionMetrics.
    """  # fmt: skip
    y_vals = training_inputs["y_vals"]
    # y_vals.shape: [batch, num_num_feats, 1]
    target_index = eval_inputs["target_index"]
    # target_index.shape: [batch, 1, 1]
    logits = torch.take_along_dim(logits, target_index, dim=-2)  # [batch, 1]
    y_vals = torch.take_along_dim(y_vals, target_index, dim=-2)  # [batch, 1]
    loss = self.loss(
        logits, training_inputs={"y_vals": y_vals, "loss_weights": 1.0}
    ).item()
    return RegressionMetrics(mse=loss)
