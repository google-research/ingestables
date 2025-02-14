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

from absl import logging
from etils import epath
import torch
from torch import nn
import tqdm
import transformers


# TODO(mononito): Maybe add interface definitions based on prior code

ROOT_DIR = "~/ingestables/"


################################################################################
####################### TOKENIZERS #############################################
################################################################################


class HuggingFaceAutoTokenizer:
  """Wrapper for AutoTokenizer from the hugging face transformers library."""

  def __init__(
      self,
      text_encoder_name: str = "st5",  # pylint: disable=unused-argument
      model_path: str = ROOT_DIR + "huggingface/sentence-t5-base",
      do_lower_case: bool = False,
      max_seq_length: int = 512,
      use_gpu: bool = True,
      use_fast: bool = True,
  ):
    """Wrapper for AutoTokenizer from the huggingface transformers library.

    Args:
      text_encoder_name: Text encoder name.
      model_path: Model path.
      do_lower_case: Do lower case.
      max_seq_length: Max sequence length.
      use_gpu: Use gpu.
      use_fast: Use fast tokenizer. If False, it will use the slow sentencepiece
        tokenizer.
    """
    if isinstance(model_path, str):
      model_path = epath.Path(model_path)
    self._tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=use_fast
    )
    self.max_seq_length = 512
    self.do_lower_case = do_lower_case
    self.max_seq_length = max_seq_length

    self.move_to_gpu = False
    if use_gpu:
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
        add_special_tokens=False,
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

  @property
  def vocab_size(self) -> int:
    """Returns the size of vocabulary of the tokenizer."""
    return self._tokenizer.vocab_size


################################################################################
####################### ENCODERS ###############################################
################################################################################


class HuggingFaceAutoModelEncoder(nn.Module):
  """AutoModels from the HuggingFace transformers library."""

  def __init__(
      self,
      text_encoder_name: str = "st5",
      model_path: str = ROOT_DIR + "huggingface/sentence-t5-base",
      use_gpu: bool = True,
  ):
    """AutoModels from the HuggingFace transformers library.

    Args:
      text_encoder_name: Text encoder name.
      model_path: Model path.
      use_gpu: Use gpu.
    """
    super().__init__()
    if isinstance(model_path, str):
      model_path = epath.Path(model_path)

    model = transformers.AutoModel.from_pretrained(
        model_path, local_files_only=True
    )

    if getattr(model.config, "is_encoder_decoder", False):
      self.model = model.get_encoder()
    else:
      self.model = model

    if use_gpu:
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


class SimpleEncoder(nn.Module):
  """Simple encoder based on TransTab."""

  def __init__(
      self,
      vocab_size: int,
      hidden_size: int = 128,
      padding_idx: int = 0,
      layer_norm_eps: float = 1e-5,
      hidden_dropout_prob: float = 0.0,
      use_gpu: bool = True,
  ):
    """Simple encoders to embed textual features.

    Args:
      vocab_size: Vocab size.
      hidden_size: Hidden size.
      padding_idx: Padding idx.
      layer_norm_eps: Layer norm eps.
      hidden_dropout_prob: Hidden dropout prob.
      use_gpu: Use gpu.
    """
    super().__init__()

    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.padding_idx = padding_idx
    self.layer_norm_eps = layer_norm_eps
    self.hidden_dropout_prob = hidden_dropout_prob

    self.model = nn.Sequential(
        nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx),
        nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps),
        nn.Dropout(self.hidden_dropout_prob),
    )

    if use_gpu:
      if self._check_if_gpu_available():
        self.model.to("cuda")
        # TODO(mononito): Maybe specify the device
      else:
        warnings.warn("GPU unavailable, using CPU instead.")

    # The embedding matrix is randomly initialized, and must be trained.
    logging.info(
        "The embedding matrix is randomly initialized, and must be trained."
    )
    self.model.train()

  @property
  def embedding_dim(self) -> int:
    """Returns the dimension of the embedding.

    Embedding size is the size of the hidden layers in the model.
    model.config.hidden_size is the size of the hidden layers in the model,
    which is typically the size of the output for transformer models.
    """
    return self.hidden_size

  @property
  def text_encoder_name(self) -> str:
    return "SimpleEncoder"

  def _check_if_gpu_available(self) -> bool:
    if torch.cuda.is_available() and torch.cuda.is_available() > 0:
      return True
    return False

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    tokens = torch.tensor(tokens)
    return self.model(tokens)


################################################################################
####################### TOKENIZER+ENCODERS #####################################
################################################################################


class _TransTabTokenizerEncoder(nn.Module):
  """TransTab Tokenizer and Model."""

  def __init__(
      self,
      text_encoder_name: str = "st5",
      model_path: str = ROOT_DIR + "huggingface/sentence-t5-base",
      do_lower_case: bool = False,
      max_seq_length: int = 512,
      use_gpu: bool = True,
      vocab_size: int | None = None,
      use_fast: bool = True,
  ):
    super().__init__()
    self.tokenizer = HuggingFaceAutoTokenizer(
        text_encoder_name,
        model_path,
        do_lower_case,
        max_seq_length,
        use_gpu,
        use_fast,
    )
    self.vocab_size = vocab_size or self.tokenizer.vocab_size
    self.encoder = SimpleEncoder(
        self.vocab_size,
    )

  def forward(
      self,
      batch: list[str] | torch.Tensor,
  ) -> torch.Tensor:

    # Tokenize the input.
    tokenized_input = self.tokenizer.tokenize(batch)

    # Compute token embeddings.
    token_embeddings = self.encoder(tokenized_input["input_ids"])

    # Account for attention mask for correct averaging.
    input_mask_expanded = torch.tile(
        tokenized_input["attention_mask"].unsqueeze(-1),
        dims=(1, 1, self.encoder.embedding_dim),
    )

    # Attention aware token embeddings
    return (token_embeddings * input_mask_expanded).cpu()


class _ST5TokenizerEncoder(nn.Module):
  """Sentence T5 Tokenizer and Model."""

  def __init__(
      self,
      text_encoder_name: str = "st5",
      model_path: str = ROOT_DIR + "huggingface/sentence-t5-base",
      do_lower_case: bool = False,
      max_seq_length: int = 512,
      use_gpu: bool = True,
      use_fast: bool = True,
  ):
    super().__init__()
    self.tokenizer = HuggingFaceAutoTokenizer(
        text_encoder_name,
        model_path,
        do_lower_case,
        max_seq_length,
        use_gpu,
        use_fast,
    )
    self.encoder = HuggingFaceAutoModelEncoder(
        text_encoder_name,
        model_path,
        use_gpu,
    )

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


class _StubTokenizerEncoder(nn.Module):
  """Stub tokenizer encoder."""

  def __init__(
      self,
      text_encoder_name: str = "stub",
      do_lower_case: bool = False,
      max_seq_length: int = 512,
      vocab_size: int | None = None,
      embedding_dim: int = 768,
      use_fast: bool = True,  # pylint: disable=unused-argument
  ):
    super().__init__()
    self.embedding_dim = embedding_dim

  def forward(
      self,
      batch: list[str] | torch.Tensor,
  ) -> torch.Tensor:
    return torch.ones(len(batch), self.embedding_dim)


class TokenizerEncoderName(enum.Enum):
  """The type of the tokenizer encoder."""

  ST5 = "st5"
  TRANSTAB = "TransTabWordEmbedding"
  STUB = "stub"


def _create_tokenizer_encoder(
    text_encoder_name: str,
    model_path: str,
    do_lower_case: bool,
    max_seq_length: int,
    use_gpu: bool,
    vocab_size: int | None = None,
    embedding_dim: int = 768,
    use_fast: bool = True,
) -> nn.Module:
  """Create text encoder from config."""
  if text_encoder_name == TokenizerEncoderName.ST5.value:
    return _ST5TokenizerEncoder(
        text_encoder_name,
        model_path,
        do_lower_case,
        max_seq_length,
        use_gpu,
        use_fast,
    )
  elif text_encoder_name == TokenizerEncoderName.TRANSTAB.value:
    return _TransTabTokenizerEncoder(
        text_encoder_name,
        model_path,
        do_lower_case,
        max_seq_length,
        use_gpu,
        vocab_size,
        use_fast,
    )
  elif text_encoder_name == TokenizerEncoderName.STUB.value:
    return _StubTokenizerEncoder(
        text_encoder_name,
        do_lower_case,
        max_seq_length,
        vocab_size,
        embedding_dim,
        use_fast,
    )
  raise ValueError(f"Unknown tokenizer encoder type: {text_encoder_name}")


class TextEncoder(nn.Module):
  """Text encoder to encoder string and categorical features."""

  def __init__(
      self,
      text_encoder_name: str = "st5",
      model_path: str = ROOT_DIR + "huggingface/sentence-t5-base",
      embedding_dim=768,
      do_lower_case: bool = False,
      max_seq_length: int = 512,
      use_gpu: bool = True,
      use_fast: bool = True,
      vocab_size: int | None = None,
      encoding_batch_size: int = 1024,
  ):
    super().__init__()
    self._text_encoder_name = text_encoder_name
    self.encoder = _create_tokenizer_encoder(
        text_encoder_name,
        model_path,
        do_lower_case,
        max_seq_length,
        use_gpu,
        vocab_size,
        embedding_dim,
        use_fast,
    )
    self._embedding_dim = embedding_dim
    self._encoding_batch_size = encoding_batch_size

  @property
  def text_encoder_name(self) -> str:
    return self._text_encoder_name

  @property
  def embedding_dim(self) -> int:
    return self._embedding_dim

  def forward(self, text_list: list[bytes | str]) -> torch.Tensor:
    if isinstance(text_list, str):
      text_list = [text_list]

    num_samples = len(text_list)
    # Encode in batches to avoid GPU OOM.
    batch = []
    for i in tqdm.trange(
        0, num_samples, self._encoding_batch_size, desc="Text encoding:"
    ):
      batch.append(
          self.encoder(
              text_list[i : min(i + self._encoding_batch_size, num_samples)]
          )
      )
    return torch.cat(batch, dim=0)


################################################################################
####################### Test Cases (Runs on Colab) #############################
################################################################################

# def test_text_encoder(self):
#     text_encoder = text_encoders.TextEncoder(
#         text_encoder_name="st5",
#         model_path=ROOT_DIR + "huggingface/sentence-t5-base",
#         do_lower_case=False,
#         max_seq_length=512,
#         use_gpu=True,
#     )

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
