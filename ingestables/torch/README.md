# IngesTables PyTorch

### Running locally

Perhaps you just added a new dataset, or tweaked some logic, and you want to
quickly (in ~ 2-5 mins) figure out whether the code would crash, and whether the
losses are going down in the first 100 steps or so. If so, you can run it on
your local development machine. It's fine to run it on one CPU; you can set
hyperparameters to make the Transformer small in order to save memory. The
training code also detects if you are running locally, and will only evaluate
the first 10 batches of each eval dataset.

#### Set up a virtual environment in your local machine (do this once)

First, create a virtual environment locally:

```shell
cd ~
python3 -m venv ingestables-venv
source ingestables-venv/bin/activate
```

Then install the requirements:

```shell
python3 -m pip install --upgrade pip
git clone ...
cd ingestables
python3 -m pip install -e .
```

You now have a virtual environment that contains the dependencies!

#### Running tests

```shell
python3 -m pip install -e .[dev]
python3 -m pytest --pyargs ingestables
```