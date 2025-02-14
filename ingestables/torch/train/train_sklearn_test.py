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

import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from ingestables.torch.data import encoders
from ingestables.torch.data import pipeline as pipeline_lib
from ingestables.torch.data import preprocessors
from ingestables.torch.data import scenario_generators
from ingestables.torch.model import head
from ingestables.torch.model import sklearn_model
from ingestables.torch.model import text_encoders
from ingestables.torch.train import metrics
from ingestables.torch.train import train_sklearn
from sklearn import linear_model


class TrainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tmp_dir)

  @parameterized.parameters(
      (linear_model.LogisticRegression,),
      # (linear_model.LinearRegression,),
  )
  def test_train_classification(self, model_cls):
    model = sklearn_model.SklearnModel(
        model_cls(), head.SklearnClassification()
    )

    pipeline = pipeline_lib.Pipeline(
        pipeline_modules=[
            pipeline_lib.PipelineModule(
                benchmark_name="test",
                dataset_name="xor",
                splitter=scenario_generators.Splitter(random_state=13),
                sampler=scenario_generators.Sampler(
                    sampling_type="full", random_state=13
                ),
                preprocessor=preprocessors.Preprocessor(),
                encoder=encoders.Encoder(
                    text_encoder=text_encoders.TextEncoder(
                        text_encoder_name="stub"
                    )
                ),
            ),
        ],
    )

    metrics_store = metrics.InMemoryMetricsStore()
    metrics_writer = metrics.MetricsWriter(metrics_store)

    trainer = train_sklearn.Trainer(
        workdir=self.tmp_dir,
        model=model,
        pipeline=pipeline,
        metrics_writer=metrics_writer,
    )
    trainer.run()

    self.assertGreater(
        metrics_store.metric(0, ("xor", "cat"), "test_accuracy"), 0.0
    )


if __name__ == "__main__":
  absltest.main()
