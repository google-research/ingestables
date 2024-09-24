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

r"""Defines the interface for tokenizers and encoders.

Plus some implementations of tokenizers and encoders are provided below.

Currently the following model types are supported:
TODO(mononito): Update this description.

  usage:
    encoder = encoders.TextEncoder(config)
"""

import copy
import enum
import warnings
from etils import epath
import ml_collections
import torch
from torch import nn
import transformers


# TODO(mononito): Maybe add interface definitions based on prior code

ROOT_DIR = "~/ingestables/"


def get_text_encoder_config() -> ml_collections.ConfigDict:
  """Text Encoder Config."""

  config = ml_collections.ConfigDict()
  config.text_encoder_name = "st5"  # "st5"
  config.model_path = ROOT_DIR + "huggingface/sentence-t5-base"
  # Model paths
  # Sentence T5:
  # Base: ROOT_DIR + "huggingface/sentence-t5-base"
  # XL: ROOT_DIR + "huggingface/sentence-t5-xl"
  # XXL: ROOT_DIR + "huggingface/sentence-t5-xxl"
  config.batch_size = 1024
  config.do_lower_case = False
  config.max_seq_length = 512
  config.use_gpu = True

  return config


################################################################################
####################### TOKENIZERS #############################################
################################################################################


class HuggingFaceAutoTokenizer:
  """Wrapper for AutoTokenizer from the hugging face transformers library."""

  def __init__(self, config: ml_collections.ConfigDict):
    """Wrapper for AutoTokenizer from the huggingface transformers library.

    Args:
      config: Text encoder config.
    """
    path = config.model_path
    if isinstance(path, str):
      path = epath.Path(path)
    self._tokenizer = transformers.AutoTokenizer.from_pretrained(
        path, local_files_only=True
    )
    self.do_lower_case, self.max_seq_length = False, 512
    if "do_lower_case" in config:
      self.do_lower_case = config.do_lower_case
    if "max_seq_length" in config:
      self.max_seq_length = config.max_seq_length

    self.move_to_gpu = False
    if config.use_gpu:
      if self._check_if_gpu_available():
        self.move_to_gpu = True
        # TODO(mononito): Maybe specify the device
      else:
        warnings.warn("GPU unavailable, using CPU instead.")

  def tokenize(
      self,
      batch: list[bytes | str] | torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Tokenizes a batch of inputs."""
    if isinstance(batch, torch.Tensor):
      batch = batch.numpy().tolist()
    if not batch:
      return {}
    if all(isinstance(b, bytes) for b in batch):
      batch = [b.decode("utf-8") for b in batch]  # pytype: disable=attribute-error

    # Strip white character space
    batch = [str(s).strip() for s in batch]

    # Lowercase
    if self.do_lower_case:
      batch = [s.lower() for s in batch]

    tokens = self._tokenizer.batch_encode_plus(
        batch,
        max_length=self.max_seq_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    if self.move_to_gpu:
      return self.move_tokens_to_gpu(tokens)
    return tokens

  def move_tokens_to_gpu(
      self, tokens: dict[str, torch.Tensor]
  ) -> dict[str, torch.Tensor]:
    tokens_in_gpu = copy.deepcopy(tokens)
    for key, value in tokens.items():
      tokens_in_gpu[key] = value.to("cuda")
      # TODO(mononito): Maybe specify gpu
    return tokens_in_gpu

  def _check_if_gpu_available(self) -> bool:
    if torch.cuda.is_available() and torch.cuda.is_available() > 0:
      return True
    return False

  def detokenize(self, ids: dict[str, torch.Tensor]) -> list[bytes | str]:
    return self._tokenizer.batch_decode(ids["input_ids"])


################################################################################
####################### ENCODERS ###############################################
################################################################################


class HuggingFaceAutoModelEncoder(nn.Module):
  """AutoModels from the HuggingFace transformers library."""

  def __init__(self, config: ml_collections.ConfigDict):
    """AutoModels from the HuggingFace transformers library.

    Args:
      config: Text encoder config.
    """
    super().__init__()
    path = config.model_path
    if isinstance(path, str):
      path = epath.Path(path)

    model = transformers.AutoModel.from_pretrained(path, local_files_only=True)

    if getattr(model.config, "is_encoder_decoder", False):
      self.model = model.get_encoder()
    else:
      self.model = model

    if config.use_gpu:
      if self._check_if_gpu_available():
        self.model.to("cuda")
        # TODO(mononito): Maybe specify the device
      else:
        warnings.warn("GPU unavailable, using CPU instead.")

    self.model.eval()  # Set model to eval mode so it doesn't track gradients

  @property
  def embedding_dim(self) -> int:
    """Returns the dimension of the embedding.

    Embedding size is the size of the hidden layers in the model.
    model.config.hidden_size is the size of the hidden layers in the model,
    which is typically the size of the output for transformer models.
    """
    return self.model.config.hidden_size

  def _check_if_gpu_available(self) -> bool:
    if torch.cuda.is_available() and torch.cuda.is_available() > 0:
      return True
    return False

  def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
      return self.model(**tokens)


################################################################################
####################### TOKENIZER+ENCODERS #####################################
################################################################################


class _ST5TokenizerEncoder(nn.Module):
  """Sentence T5 Tokenizer and Model."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.tokenizer = HuggingFaceAutoTokenizer(config)
    self.encoder = HuggingFaceAutoModelEncoder(config)

  def forward(
      self,
      batch: list[str] | torch.Tensor,
  ) -> torch.Tensor:
    """Embeds a single batch of texts with the given model and tokenizer."""
    # Tokenize the input.
    tokenized_input = self.tokenizer.tokenize(batch)

    # Compute token embeddings.
    model_output = self.encoder(tokenized_input)

    token_embeddings = model_output.last_hidden_state  # pylint: disable=attribute-error

    # Account for attention mask for correct averaging.
    input_mask_expanded = torch.tile(
        tokenized_input["attention_mask"].unsqueeze(-1),
        dims=(1, 1, self.encoder.embedding_dim),
    )

    # Sum the embeddings after multiplying by the attention mask.
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

    # Get the denominator for the mean.
    sum_mask = torch.clamp(
        torch.sum(input_mask_expanded, dim=1),
        min=1e-9,
        max=float("inf"),
    )

    # Compute the mean.
    # Note: because Sentence T5 embeddings are trained with mean pooling
    # This might need to be modified for other models.
    sentence_embeddings = sum_embeddings / sum_mask
    sentence_embeddings = torch.nn.functional.normalize(
        sentence_embeddings, p=2, dim=1
    )

    return sentence_embeddings.detach().cpu()


class TokenizerEncoderName(enum.Enum):
  """The type of the tokenizer encoder."""

  ST5 = "st5"


def _create_tokenizer_encoder(config: ml_collections.ConfigDict) -> nn.Module:
  """Create text encoder from config."""
  if config.text_encoder_name == TokenizerEncoderName.ST5.value:
    return _ST5TokenizerEncoder(config)
  raise ValueError(
      f"Unknown tokenizer encoder type: {config.text_encoder_name}"
  )


class TextEncoder(nn.Module):
  """Text encoder to encoder string and categorical features."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__()
    self.encoder = _create_tokenizer_encoder(config)
    # TODO(mononito): Implement support for caching

  @property
  def embedding_dim(self) -> int:
    return self.encoder.embedding_dim

  def forward(
      self, text_list: list[bytes | str] | torch.Tensor
  ) -> torch.Tensor:
    return self.encoder(text_list)


################################################################################
####################### Test Cases (Runs on Colab) #############################
################################################################################

# def test_text_encoder(self):
#     text_encoder_cfg = text_encoders.get_text_encoder_config()
#     text_encoder = text_encoders.TextEncoder(text_encoder_cfg)

#     text_list = [
#         "What is the capital city of USA?",
#         "Washington DC",
#         "New York City",
#         "Seattle",
#         "Which direction does the sun rise?",
#         "East",
#         "West",
#         "North",
#         "Sorth",
#     ]

#     encoded_outputs = text_encoder(text_list)
#     assert
#         torch.argmax(
#             torch.einsum("kd,d->k", encoded_outputs[:4], encoded_outputs[4])
#         ) == 0
#     assert
#         torch.argmax(
#             torch.einsum("kd,d->k", encoded_outputs[5:], encoded_outputs[4])
#         ) == 0
