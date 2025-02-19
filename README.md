# IngesTables: Scalable and Efficient Training of LLM-Enabled Tabular Foundation Models

[![Unittests](https://github.com/google-research/ingestables/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/google-research/ingestables/actions/workflows/pytest_and_autopublish.yml)
[![PyPI version](https://badge.fury.io/py/ingestables.svg)](https://badge.fury.io/py/ingestables)

This repository contains the implementation of IngesTables, a tabular foundation 
model. An earlier version of the model was accepted at the NeurIPS'23 [Tabular 
Representation Learning Workshop](https://neurips.cc/virtual/2023/81311). Stay 
tuned for updates!

It contains library code that defines the data preprocessing, training, and
evaluation. It also contains scripts for running it locally or on Google Cloud
Platform (GCP). Do note that using the GCP scripts may incur costs and would
transmit data to GCP and be accessible to those who can access your GCP project.

## 🧑‍🏫 Tutorials

Here is the list of tutorials and reproducibile experiments to get started with 
IngesTables for various tasks:
- [Regression]()
- [Classification]()

## BibTeX

If you found any part of this codebase to be useful, please consider citing 
our work:

```bibtex
@inproceedings{yak2023ingestables,
  title={IngesTables: Scalable and Efficient Training of LLM-Enabled Tabular Foundation Models},
  author={Scott Yak and Yihe Dong and Javier Gonzalvo and Sercan Arik},
  booktitle={NeurIPS'23 Table Representation Learning Workshop},
  year={2023}
}
```

*This is not an officially supported Google product.*
