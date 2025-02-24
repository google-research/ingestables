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

import dataclasses

from absl.testing import absltest
from ingestables.torch import types
from ingestables.torch.train import metrics as metrics_lib


_FAKE_HEAD_KEY = "fake"


@dataclasses.dataclass
class FakeMetrics(types.Metrics):
  """Fake metrics for testing.

  Attributes:
    metric_a: Fake metric A.
    metric_b: Fake metric B.
    metric_c: Optional fake metric C.
  """

  metric_a: float
  metric_b: float
  metric_c: float | None = None

  def metrics_dict(self) -> dict[str, float | None]:
    return dataclasses.asdict(self)

  @property
  def head_key(self) -> str:
    return _FAKE_HEAD_KEY


class InMemoryMetricsStoreTest(absltest.TestCase):

  def test_write_metrics(self):
    store = metrics_lib.InMemoryMetricsStore()

    step = 10
    key = ("dataset1", "head1")
    metrics = {"metric1": 0.8, "metric2": 0.9}

    store.write_metrics(step, key, metrics)

    self.assertEqual(store.metrics, {step: {key: metrics}})

    step2 = 20
    key2 = key
    metrics2 = {"metric1": 0.4, "metric2": 0.3}

    store.write_metrics(step2, key2, metrics2)
    # Check that the metrics are correctly appended.
    self.assertEqual(
        store.metrics, {step: {key: metrics}, step2: {key2: metrics2}}
    )


class MetricsWriterTest(absltest.TestCase):

  def test_write_metrics(self):
    store = metrics_lib.InMemoryMetricsStore()
    writer = metrics_lib.MetricsWriter(store)

    step = 1
    key = ("dataset1", "head1")
    metrics = {"metric1": 0.8, "metric2": 0.9}

    writer.write_metrics(step, key, metrics)

    self.assertEqual(store.metrics, {step: {key: metrics}})

  def test_write_metric(self):
    store = metrics_lib.InMemoryMetricsStore()
    writer = metrics_lib.MetricsWriter(store)

    step = 5
    key = ("dataset2", "head2")
    name = "single_metric"
    value = 0.75

    writer.write_metric(step, key, name, value)

    self.assertEqual(store.metrics, {step: {key: {name: value}}})

  def test_write_model_metrics(self):
    store = metrics_lib.InMemoryMetricsStore()
    writer = metrics_lib.MetricsWriter(store)

    step = 2
    dataset_type = "train"
    dataset_key = "datasetA"
    head_key = _FAKE_HEAD_KEY
    metrics_dict_val = {"metric_a": 0.9, "metric_b": 0.8, "metric_c": None}
    fake_metrics = FakeMetrics(**metrics_dict_val)

    expected_metrics = {
        f"{dataset_type}_metric_a": 0.9,
        f"{dataset_type}_metric_b": 0.8,
    }  # metric_c should be filtered out

    writer.write_model_metrics(step, dataset_type, dataset_key, fake_metrics)

    self.assertEqual(
        store.metrics,
        {
            step: {
                (dataset_key, head_key): expected_metrics,
            },
        },
    )


if __name__ == "__main__":
  absltest.main()
