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

from absl import logging
from absl.testing import absltest
from ingestables.torch.data import encoders
from ingestables.torch.data import pipeline as pipeline_lib
from ingestables.torch.data import preprocessors
from ingestables.torch.data import scenario_generators
from ingestables.torch.model import text_encoders


class PipelineTest(absltest.TestCase):

  def test_pipeline(self):
    pipeline = pipeline_lib.Pipeline(
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
    )

    xor_train_data = pipeline.get_train_data("xor")
    logging.info("xor_train_data[0:4]: %s", xor_train_data[0:4])
    xor_val_data = pipeline.get_val_data("xor")
    logging.info("xor_val_data[0:4]: %s", xor_val_data[0:4])
    xor_test_data = pipeline.get_test_data("xor")
    logging.info("xor_test_data[0:4]: %s", xor_test_data[0:4])

    and_train_data = pipeline.get_train_data("and")
    logging.info("and_train_data[0:4]: %s", and_train_data[0:4])
    and_val_data = pipeline.get_val_data("and")
    logging.info("and_val_data[0:4]: %s", and_val_data[0:4])
    and_test_data = pipeline.get_test_data("and")
    logging.info("and_test_data[0:4]: %s", and_test_data[0:4])


if __name__ == "__main__":
  absltest.main()
