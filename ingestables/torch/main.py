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

r"""ENTRYPOINT for running on PyTorch with GPUs.

Usage:

_FIDDLE_CONFIG = absl_flags.DEFINE_fiddle_config(
    "fiddle_config",
    default=None,
    help_string="Basic config for ingestables.Model.",
    default_module=basic_config,
)


def main(argv):
  del argv
  logging.info("Job started")

  cfg = _FIDDLE_CONFIG.value

  logging.info(
      "\n\n########## BEGIN FIDDLE CONFIG ##########\n"
      "%s"
      "\n########## END FIDDLE CONFIG ##########\n\n",
      printing.as_str_flattened(cfg),  # pytype: disable=attribute-error
  )

  tfl: tabular_foundation_lab.TablularFoundationLab = fdl.build(cfg)
  tfl.run()


if __name__ == "__main__":
  app.run(main, flags_parser=absl_flags.flags_parser)
