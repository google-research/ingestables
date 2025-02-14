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

"""Utility functions for loading and pre-processing datasets."""

import copy
import json
from typing import List, Literal, Tuple
import warnings

from absl import logging
import arff
from etils import epath
from ingestables.torch import types
from ingestables.torch.data import preprocessing
import numpy as np
import pandas as pd
import scipy.io.arff as scipy_load


ROOT_PATH = epath.Path("~/ingestables")
BASE_PATHS = {
    "carte": ROOT_PATH / "datasets/carte/preprocessed",
    "opentabs": ROOT_PATH / "datasets/opentabs",
    "ingestables": ROOT_PATH / "datasets/verticals/processed",
}

# [NOTE] Some datasets can have multiple targets. For example, the
# `employee_remuneration` dataset has two targets:
# "salary_increase_percentage" and "salary_increase_amount". We should be
# careful when using these datasets because there may be leakage.
APPROVED_CARTE_DATASETS = [
    "anime_planet",
    "babies_r_us",
    "beer_ratings",
    "bikedekho",
    "bikewale",
    "buy_buy_baby",
    "cardekho",
    "chocolate_bar_ratings",
    "clear_corpus",
    "employee_remuneration",  # [NOTE] Same dataset, different target
    "employee_salaries",  # [NOTE] Same dataset, different target
    "filmtv_movies",
    "jp_anime",
    "michelin",
    "movies",
    "nba_draft",
    "prescription_drugs",
    "rotten_tomatoes",
    "spotify",
    "us_accidents_counts",  # [NOTE] Same dataset, different target
    "us_accidents_severity",  # [NOTE] Same dataset, different target
    "used_cars_pakistan",
    "videogame_sales",
    "wikiliq_beer",  # [NOTE] Same dataset, different target
    "wikiliq_spirit",  # [NOTE] Same dataset, different target
    "wina_pl",
    "wine_dot_com_prices",  # [NOTE] Same dataset, different target
    "wine_dot_com_ratings",  # [NOTE] Same dataset, different target
    "wine_enthusiasts_prices",  # [NOTE] Same dataset, different target
    "wine_enthusiasts_ratings",  # [NOTE] Same dataset, different target
    "yelp",
    "zomato",
]

# [NOTE] The following datasets have unknown license. We should be careful when
# using these datasets.
CARTE_DATASETS_UNKNOWN_LICENSE = [
    "coffee_ratings",
    "company_employees",
    "journal_jcr",
    "mlds_salaries",
    "museums",
    "used_cars_saudi_arabia",
    "us_presidential",
    "used_cars_24",
    "used_cars_benz_italy",
    "whisky",
]

CARTE_REGRESSION_DATASETS = [
    "anime_planet",
    "babies_r_us",
    "beer_ratings",
    "bikedekho",
    "bikewale",
    "buy_buy_baby",
    "cardekho",
    "clear_corpus",
    "employee_remuneration",
    "employee_salaries",
    "filmtv_movies",
    "jp_anime",
    "movies",
    "prescription_drugs",
    "rotten_tomatoes",
    "us_accidents_counts",
    "used_cars_pakistan",
    "videogame_sales",
    "wikiliq_beer",
    "wikiliq_spirit",
    "wina_pl",
    "wine_dot_com_prices",
    "wine_dot_com_ratings",
    "wine_enthusiasts_prices",
    "wine_enthusiasts_ratings",
]
CARTE_CLASSIFICATION_DATASETS = [
    "chocolate_bar_ratings",
    "michelin",
    "nba_draft",
    "spotify",
    "us_accidents_severity",
    "yelp",
    "zomato",
    "coffee_ratings",
    "whisky",
]
INGESTABLES_REGRESSION_DATASETS = [
    "nyc_housing",
    "us_real_estate",
    "usa_housing",
    "us_airbnb",
    "nashville_housing",
]
INGESTABLES_CLASSIFICATION_DATASETS = [
    "autos",
    "home_credit",
    "give_me_some_credit",
    "south_africa_debt",
    "indonesian_telecom_delinquency",
]


# --------------------------------------------------------------------------
# Dataset Loading Functions
# --------------------------------------------------------------------------

XorDataFrame = pd.DataFrame({
    "x1": np.array([0.0, 0, 1, 1] * 1000),
    "x2": np.array([0.0, 1, 0, 1] * 1000),
    "y": np.array(["0", "1", "1", "0"] * 1000),
})

AndDataFrame = pd.DataFrame({
    "x1": [0, 0, 1, 1] * 1000,
    "x2": [0, 1, 0, 1] * 1000,
    "y": ["1", "0", "0", "1"] * 1000,
})

StocksDataFrame = pd.DataFrame({
    "x": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 1000),
    "y": np.array(
        [
            180.0,
            181.0,
            182.0,
            183.0,
            184.0,
            185.0,
            186.0,
            187.0,
            188.0,
            189.0,
        ]
        * 1000
    ),
})


def load_test_dataset(
    dataset_name: str,
) -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Load a *TEST** dataset."""
  if dataset_name == "xor":
    return XorDataFrame, types.create_task_info(
        task_type="classification",
        target_key="y",
        target_classes=["0", "1"],
        dataset_name="xor",
    )
  elif dataset_name == "and":
    return AndDataFrame, types.create_task_info(
        task_type="classification",
        target_key="y",
        target_classes=["0", "1"],
        dataset_name="and",
    )
  elif dataset_name == "stocks":
    return StocksDataFrame, types.create_task_info(
        task_type="regression",
        target_key="y",
        dataset_name="stocks",
    )
  else:
    raise ValueError(f"Unknown dataset {dataset_name}")


def load_carte_dataset(
    dataset_name: str,
) -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Load a pre-processed CARTE dataset."""

  if dataset_name in CARTE_DATASETS_UNKNOWN_LICENSE:
    warnings.warn(
        "Loading a dataset with unknown license. Exercise caution when using"
        + " these datasets.",
        UserWarning,
    )

  df_path = BASE_PATHS["carte"] / dataset_name / "raw.parquet"
  config_path = BASE_PATHS["carte"] / dataset_name / "config_data.json"

  with df_path.open("rb") as f:
    data = pd.read_parquet(f)

  with config_path.open("rb") as f:
    config = json.load(f)

  # Do a quick type check of the target value
  target_key = config["target_name"]
  task = config["task"]

  target_dtype = data.dtypes.to_dict()[target_key]
  if task == "regression":
    assert target_dtype == "float"
  elif task == "classification":
    if target_dtype not in ["object", "string"]:
      warnings.warn(
          "Classification target should be either object or string, but got"
          f" {target_dtype}"
      )

    if target_dtype in ["float", "int"]:
      boolean_target = (data[target_key].nunique() == 2) and (
          set(pd.unique(data[target_key]).astype(int)) == set([0, 1])
      )
      if boolean_target:
        data[target_key] = data[target_key].replace({1.0: "true", 0.0: "false"})
        logging.info("Replacing boolean target to strings")

  target_key = config["target_name"]

  target_classes = pd.unique(data.loc[:, target_key]).tolist()
  if np.nan in target_classes:
    target_classes.remove(np.nan)
  if None in target_classes:
    target_classes.remove(None)

  task_information = types.create_task_info(
      task_type=task,
      target_key=target_key,
      target_classes=target_classes,
      dataset_name=dataset_name,
  )

  return data, task_information


def load_ingestables_dataset(
    dataset_name: str,
) -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Load a pre-processed Ingestables dataset."""
  return ingestables_dataset_name_to_class[dataset_name]()


benchmark_name_to_class = {
    "carte": load_carte_dataset,
    "ingestables": load_ingestables_dataset,
    "test": load_test_dataset,
}


def load_dataset_from_benchmark(
    benchmark_name: str | Literal["carte", "ingestables"],
    dataset_name: str,
) -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Load a dataset from a benchmark."""
  return benchmark_name_to_class[benchmark_name](dataset_name)


def get_dataset_names_in_benchmark(
    benchmark_name: Literal["carte", "ingestables"],
    approved_datasets_only: bool = False,
    problem_type: Literal["all", "regression", "classification"] = "all",
) -> List[str]:
  """Get dataset names in a benchmark."""
  if benchmark_name == "carte":
    carte_datasets = copy.deepcopy(APPROVED_CARTE_DATASETS)
    if not approved_datasets_only:
      carte_datasets += CARTE_DATASETS_UNKNOWN_LICENSE
    if problem_type == "regression":
      carte_datasets = [
          dataset
          for dataset in carte_datasets
          if dataset in CARTE_REGRESSION_DATASETS
      ]
    elif problem_type == "classification":
      carte_datasets = [
          dataset
          for dataset in carte_datasets
          if dataset in CARTE_CLASSIFICATION_DATASETS
      ]
    return carte_datasets
  elif benchmark_name == "ingestables":
    risk_dataset_names = [
        "autos",
        "home_credit",
        "give_me_some_credit",
        "south_africa_debt",
        "indonesian_telecom_delinquency",
    ]
    real_estate_dataset_names = [
        "nyc_housing",
        "us_real_estate",
        "usa_housing",
        "us_airbnb",
        "nashville_housing",
    ]
    if problem_type == "regression":
      return real_estate_dataset_names
    elif problem_type == "classification":
      return risk_dataset_names
    else:
      return risk_dataset_names + real_estate_dataset_names

  else:
    raise ValueError(f"Unknown benchmark {benchmark_name}")


# --------------------------------------------------------------------------
# Data reading functions.
# --------------------------------------------------------------------------


def read_arff(
    vertical_name: str = "00_risk",
    dataset_name: str = "04_south_africa_debt",
) -> tuple[np.ndarray, scipy_load.MetaData]:
  """Function to read arff dataset."""

  path = (
      BASE_PATHS["ingestables"].parent
      / vertical_name
      / dataset_name
      / "raw_dataset.arff"
  )

  try:
    with path.open(mode="r") as f:
      data, meta = scipy_load.loadarff(f)
  except NotImplementedError as exp:
    with path.open(mode="r") as f:
      print(f"Failed to load: {exp}. Trying to load with arff.load(...)")

      del f
      with path.open(mode="r") as f:
        data_and_metadata = arff.load(f)

      attr = []
      # TODO(mononito): Import from scipy_load.arff.to_attribute does not work
      # but arffread is going to be deprecated.
      for attr_name, attr_type in data_and_metadata["attributes"]:
        attr.append(scipy_load.arffread.to_attribute(attr_name, attr_type))

      meta = scipy_load.MetaData(rel=data_and_metadata["relation"], attr=attr)

      data = []
      for i in range(len(data_and_metadata["data"])):
        data.append(
            tuple([data_and_metadata["data"][i][j] for j in range(len(attr))])
        )

      data = np.array(data, dtype=([(a.name, a.dtype) for a in attr]))

  return data, meta


# --------------------------------------------------------------------------
# Loading functions for specific Ingestables datasets.
# --------------------------------------------------------------------------


def load_autos() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed Autos dataset."""
  # Source: https://www.kaggle.com/datasets/toramky/automobile-dataset

  processed_data_path = BASE_PATHS["ingestables"] / "autos.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)

  else:
    logging.info("Preprocessing Autos dataset...")
    data, _ = read_arff(vertical_name="00_risk", dataset_name="00_autos")

    df = pd.DataFrame(data)

    # Format columns
    original_columns = list(df.columns)
    formatted_column_name = [
        " ".join(i.split("-")).title() for i in original_columns
    ]
    df.rename(
        columns=dict(zip(original_columns, formatted_column_name)), inplace=True
    )
    df.rename(columns={"Class": "Insurance Ratings"}, inplace=True)

    # Format column values
    col_to_type_map = df.dtypes.to_dict()
    str_feat_types = [
        i for i in col_to_type_map.keys() if col_to_type_map[i] == "object"
    ]
    # Change from byte strings to strings
    for col in str_feat_types:
      df[col] = df[col].astype(np.str_)

    rep_str = {
        "std": "standard",
        "turbo": "turbo charged",
        "rwd": "rear wheel drive",
        "fwd": "front wheel drive",
        "4wd": "four wheel drive",
        "dohc": "dual overhead camshaft",
        "dohcv": "dual overhead camshaft and valve",
        "ohc": "overhead camshaft",
        "ohcv": "overhead camshaft and valve",
        "ohcf": "overhead cam and valve f engine",
        "rotor": "rotary engine",
        "2bbl": "two barrel carburetor",
        "4bbl": "four barrel carburetor",
        "idi": "indirect injection",
        "mfi": "multi-port fuel injection",
        "mpfi": "multi-point fuel injection",
        "spfi": "sequential port fuel injection",
    }
    df.replace(to_replace=rep_str, inplace=True)

    insurance_ratings = {
        -3: "Very Safe",
        -2: "Safe",
        -1: "Slightly Safe",
        0: "Neutral",
        1: "Slightly Risky",
        2: "Risky",
        3: "Very Risky",
    }
    df.loc[:, "Insurance Ratings"].replace(
        to_replace=insurance_ratings, inplace=True
    )

    # Format Target
    # NOTE: Both normalized losses and the insurance rating can act as targets
    df["Insurance Ratings (Binary)"] = [
        "Safe" if "Risky" not in rating else "Risky"
        for rating in list(df["Insurance Ratings"])
    ]
    # df["Insurance Ratings (Binary)"] = ["Risky" if "Safe" not in rating else "Safe" for rating in list(df["Insurance Ratings"])]  pylint: disable=line-too-long

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  target_key = "Insurance Ratings (Binary)"
  dataset_name = "autos"
  target_classes = pd.unique(df.loc[:, target_key])
  if np.nan in target_classes:
    target_classes.remove(np.nan)
  if None in target_classes:
    target_classes.remove(None)

  task_information = types.create_task_info(
      task_type="classification",
      target_key=target_key,
      target_classes=target_classes,
      dataset_name=dataset_name,
  )

  return df, task_information


def load_home_credit() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed Home Credit dataset."""
  # Source: https://www.kaggle.com/competitions/home-credit-default-risk/data

  processed_data_path = BASE_PATHS["ingestables"] / "home_credit.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing Home Credit dataset...")
    data, _ = read_arff(vertical_name="00_risk", dataset_name="02_home_credit")

    df = pd.DataFrame(data)
    df.drop(columns=["FLAG_DOCUMENT_2"], inplace=True)  # Only 1 unique value

    # Format column values
    col_to_type_map = df.dtypes.to_dict()
    str_feat_types = [
        i for i in col_to_type_map.keys() if col_to_type_map[i] == "object"
    ]
    # Change from byte strings to strings
    for col in str_feat_types:
      df[col] = df[col].astype(np.str_)

    # Format Flag Columns
    original_columns = list(df.columns)
    flag_cols = [i for i in original_columns if i.startswith("FLAG_")]
    rep_str = {"1": "yes", "Y": "yes", "0": "no", "N": "no"}
    df.loc[:, flag_cols] = (
        df.loc[:, flag_cols]
        .astype(np.str_)
        .replace(to_replace=rep_str, inplace=False)
    )

    reg_rating_cols = ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]
    df.loc[:, reg_rating_cols] = df.loc[:, reg_rating_cols].astype(np.str_)

    city_related_cols = [
        "LIVE_CITY_NOT_WORK_CITY",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
    ]
    rep_str = {"1": "different", "0": "same"}
    df.loc[:, city_related_cols] = (
        df.loc[:, city_related_cols]
        .astype(np.str_)
        .replace(to_replace=rep_str, inplace=False)
    )
    rep_str = {"M": "male", "F": "female", "XNA": "N/A"}
    df.loc[:, "CODE_GENDER"] = (
        df.loc[:, "CODE_GENDER"]
        .astype(np.str_)
        .replace(to_replace=rep_str, inplace=False)
    )

    # Format columns
    col_desc_path = ROOT_PATH / (
        "datasets/verticals/00_risk/02_home_credit/column_description_cleaned.csv"
    )
    with col_desc_path.open("r") as f:
      col_desc = pd.read_csv(f)

    col_name_to_desc = col_desc.set_index("Feature Name").to_dict()[
        "Feature Description"
    ]
    df.rename(columns=col_name_to_desc, inplace=True)

    # Format column values
    rep_str = {
        "1": "Client had late payment",
        "0": "Client paid in time / other cases",
    }
    df.loc[:, "class"] = (
        df.loc[:, "class"]
        .astype(int)
        .astype(np.str_)
        .replace(to_replace=rep_str, inplace=False)
    )

    df.rename(columns={"class": "Consumer repays the loan"}, inplace=True)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  target_key = "Consumer repays the loan"
  dataset_name = "Home Credit"
  target_classes = pd.unique(df.loc[:, target_key]).tolist()
  if np.nan in target_classes:
    target_classes.remove(np.nan)
  if None in target_classes:
    target_classes.remove(None)

  task_information = types.create_task_info(
      task_type="classification",
      target_key=target_key,
      target_classes=target_classes,
      dataset_name=dataset_name,
  )

  return df, task_information


def load_give_me_some_credit() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed Give Me Some Credit dataset."""
  # Source: https://www.kaggle.com/competitions/GiveMeSomeCredit/data

  processed_data_path = BASE_PATHS["ingestables"] / "give_me_some_credit.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing Give Me Some Credit dataset...")
    data, _ = read_arff(
        vertical_name="00_risk", dataset_name="03_give_me_some_credit"
    )

    df = pd.DataFrame(data)

    # Format column values
    rep_str = {"1": "yes", "0": "no"}
    df.loc[:, "SeriousDlqin2yrs"] = (
        df.loc[:, "SeriousDlqin2yrs"]
        .astype(np.str_)
        .replace(to_replace=rep_str, inplace=False)
    )

    col_name_to_desc = {
        "SeriousDlqin2yrs": "Person experienced 90 days past due delinquency",
        "RevolvingUtilizationOfUnsecuredLines": (
            "Total balance on credit cards and personal lines of credit divided"
            + " by the sum of credit limits"
        ),
        "age": "Age of borrower in years",
        "NumberOfTime30-59DaysPastDueNotWorse": (
            "Number of times borrower has been 30-59 days past due but no worse"
            + " in the last 2 years"
        ),
        "DebtRatio": (
            "Monthly debt payments, alimony, living costs divided by monthy"
            " gross"
            + " income"
        ),
        "MonthlyIncome": "Monthly income",
        "NumberOfOpenCreditLinesAndLoans": (
            "Number of Open loans (e.g. car loan or mortgage) and Lines of"
            " credit"
            + " (e.g. credit cards)"
        ),
        "NumberOfTimes90DaysLate": (
            "Number of times borrower has been 90 days or more past due."
        ),
        "NumberRealEstateLoansOrLines": (
            "Number of mortgage and real estate loans including home equity"
            " lines"
            + " of credit"
        ),
        "NumberOfTime60-89DaysPastDueNotWorse": (
            "Number of times borrower has been 60-89 days past due but no worse"
            + " in the last 2 years."
        ),
        "NumberOfDependents": (
            "Number of dependents in family (spouse, children etc.)"
        ),
    }
    df.rename(columns=col_name_to_desc, inplace=True)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  target_key = "Person experienced 90 days past due delinquency"
  dataset_name = "Give Me Some Credit"
  target_classes = pd.unique(df.loc[:, target_key])
  if np.nan in target_classes:
    target_classes.remove(np.nan)
  if None in target_classes:
    target_classes.remove(None)

  task_information = types.create_task_info(
      task_type="classification",
      target_key=target_key,
      target_classes=target_classes,
      dataset_name=dataset_name,
  )

  return df, task_information


def load_south_africa_debt() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed Municipal Debt Risk Analysis dataset."""
  # Source: https://www.kaggle.com/datasets/dmsconsultingsa/municipal-debt-risk-analysis  pylint: disable=line-too-long

  processed_data_path = BASE_PATHS["ingestables"] / "south_africa_debt.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing South Africa Debt dataset...")

    data, _ = read_arff(
        vertical_name="00_risk", dataset_name="04_south_africa_debt"
    )

    df = pd.DataFrame(data)
    df.drop(
        columns=["accountcategoryid", "acccatabbr"], inplace=True
    )  # Redundant

    # Format values
    bin_feats = ["hasidno", "baddebt"]
    rep_str = {1.0: "yes", 0.0: "no"}
    df.loc[:, bin_feats] = df.loc[:, bin_feats].replace(
        to_replace=rep_str, inplace=False
    )

    col_name_to_desc = {
        "accountcategory": "Type of Account",
        "propertyvalue": "Market value of property",
        "propertysize": "Property Size in square metres",
        "totalbilling": "Total amount billed to the account for all services",
        "avgbilling": "Average amount billed to the account for all services",
        "totalreceipting": (
            "Total amount receipted to the account for all services"
        ),
        "avgreceipting": (
            "Average amount receipted to the account for all services"
        ),
        "total90debt": "Total Debt",
        "totalwriteoff": "Total amount of debt that has been written off",
        "collectionratio": (
            "Ratio between the total amount receipted and total billing amount"
        ),
        "debtbillingratio": (
            "Ratio between the total debt and total billing amount"
        ),
        "totalelecbill": "Total Electricity Bill",
        "hasidno": "Consumer has an ID number",
        "baddebt": "Bad Debt",
    }
    df.rename(columns=col_name_to_desc, inplace=True)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  target_key = "Bad Debt"
  dataset_name = "South Africa Debt"
  target_classes = pd.unique(df.loc[:, target_key])
  if np.nan in target_classes:
    target_classes.remove(np.nan)
  if None in target_classes:
    target_classes.remove(None)

  task_information = types.create_task_info(
      task_type="classification",
      target_key=target_key,
      target_classes=target_classes,
      dataset_name=dataset_name,
  )

  return df, task_information


def load_indonesian_telecom_delinquency() -> (
    Tuple[pd.DataFrame, types.TaskInfo]
):
  """Loads the pre-proprocessed Indonesian Telecom Delinquency dataset."""
  # Source: https://www.kaggle.com/datasets/dmsconsultingsa/municipal-debt-risk-analysis  pylint: disable=line-too-long
  # Resources: https://github.com/thamizhdatatrained/Micro-Credit-Loan-Defaulter-Project/tree/main  pylint: disable=line-too-long

  processed_data_path = (
      BASE_PATHS["ingestables"] / "indonesian_telecom_delinquency.csv"
  )

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing Indonesian Telecom Delinquency dataset...")

    path = ROOT_PATH / (
        "datasets/verticals/"
        "00_risk/05_indonesian_telecom_delinquency/raw_dataset.csv"
    )
    with path.open("r") as f:
      df = pd.read_csv(f)

    # Drop redundant / useless columns
    df.drop(columns=["msisdn", "pcircle", "pdate"], inplace=True)

    # Format values
    rep_str = {1: "yes", 0: "no"}
    df.loc[:, "label"] = df.loc[:, "label"].replace(
        to_replace=rep_str, inplace=False
    )

    col_name_to_desc = {
        "label": (
            "User paid back the credit amount within 5 days of issuing the loan"
        ),
        "aon": "Age on cellular network in days",
        "daily_decr30": (
            "Daily amount spent from main account, averaged over last 30"
            " days (in"
            + " Indonesian Rupiah)"
        ),
        "daily_decr90": (
            "Daily amount spent from main account, averaged over last 90"
            " days (in"
            + " Indonesian Rupiah)"
        ),
        "rental30": "Average main account balance over last 30 days",
        "rental90": "Average main account balance over last 90 days",
        "last_rech_date_ma": (
            "Number of days till last recharge of main account"
        ),
        "last_rech_date_da": (
            "Number of days till last recharge of data account"
        ),
        "last_rech_amt_ma": "Amount of last recharge of main account",
        "cnt_ma_rech30": (
            "Number of times main account got recharged in last 30 days"
        ),
        "fr_ma_rech30": "Frequency of main account recharged in last 30 days",
        "sumamnt_ma_rech30": (
            "Total amount of recharge in main account over last 30 days (in"
            + " Indonesian Rupiah)"
        ),
        "medianamnt_ma_rech30": (
            "Median of amount of recharges done in main account over last 30"
            " days"
            + " at user level (in Indonesian Rupiah)"
        ),
        "medianmarechprebal30": (
            "Median of main account balance just before recharge in last 30"
            " days"
            + " at user level (in Indonesian Rupiah)"
        ),
        "cnt_ma_rech90": (
            "Number of times main account got recharged in last 90 days"
        ),
        "fr_ma_rech90": "Frequency of main account recharged in last 90 days",
        "sumamnt_ma_rech90": (
            "Total amount of recharge in main account over last 90 days (in"
            + " Indonesian Rupiah)"
        ),
        "medianamnt_ma_rech90": (
            "Median of amount of recharges done in main account over last 90"
            " days"
            + " at user level (in Indonesian Rupiah)"
        ),
        "medianmarechprebal90": (
            "Median of main account balance just before recharge in last 90"
            " days"
            + " at user level (in Indonesian Rupiah)"
        ),
        "cnt_da_rech30": (
            "Number of times data account got recharged in last 30 days"
        ),
        "fr_da_rech30": "Frequency of data account recharged in last 30 days",
        "cnt_da_rech90": (
            "Number of times data account got recharged in last 90 days"
        ),
        "fr_da_rech90": "Frequency of data account recharged in last 90 days",
        "cnt_loans30": "Number of loans taken by user in last 30 days",
        "amnt_loans30": "Total amount of loans taken by user in last 30 days",
        "maxamnt_loans30": (
            "Maximum amount of loan taken by the user in last 30 days"
        ),
        "medianamnt_loans30": (
            "Median of amounts of loans taken by the user in last 30 days"
        ),
        "cnt_loans90": "Number of loans taken by user in last 90 days",
        "amnt_loans90": "Total amount of loans taken by user in last 90 days",
        "maxamnt_loans90": (
            "Maximum amount of loan taken by the user in last 90 days"
        ),
        "medianamnt_loans90": (
            "Median of amounts of loans taken by the user in last 90 days"
        ),
        "payback30": "Average payback time in days over last 30 days",
        "payback90": "Average payback time in days over last 90 days",
        "pcircle": "Telecom circle",
    }
    df.rename(columns=col_name_to_desc, inplace=True)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  target_key = (
      "User paid back the credit amount within 5 days of issuing the loan"
  )
  dataset_name = "Indonesian Telecom Delinquency"
  target_classes = pd.unique(df.loc[:, target_key])
  if np.nan in target_classes:
    target_classes.remove(np.nan)
  if None in target_classes:
    target_classes.remove(None)

  task_information = types.create_task_info(
      task_type="classification",
      target_key=target_key,
      target_classes=target_classes,
      dataset_name=dataset_name,
  )

  return df, task_information


def load_us_airbnb() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed US AirBnB dataset."""
  # Source: https://www.kaggle.com/datasets/kritikseth/us-airbnb-open-data

  processed_data_path = BASE_PATHS["ingestables"] / "us_airbnb.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing US AirBnB dataset...")
    data, _ = read_arff(
        vertical_name="01_real_estate", dataset_name="03_us_airbnb"
    )
    df = pd.DataFrame(data)

    # Drop redundant, useless, ID-type and missing features
    df.drop(
        columns=[
            "id",
            "host_name",
            "host_id",
            "last_review",
            "neighbourhood_group",
        ],
        inplace=True,
        errors="ignore",
    )

    # Format column values
    repr_str = {None: "No name or description"}
    df["name"] = df["name"].replace(to_replace=repr_str)
    repr_str = {np.NaN: 0}  # Replace missing reviews per month to 0
    df["reviews_per_month"] = df["reviews_per_month"].replace(
        to_replace=repr_str
    )
    repr_str = {"Entire home/apt": "Entire home or apartment"}
    df["room_type"] = df["room_type"].replace(to_replace=repr_str)
    # Clean the name column
    df["name"] = df["name"].map(
        lambda x: preprocessing.clean_text(text=x, truncate_len=128)
    )

    # Format column name
    col_name_to_desc = {
        "name": "Name",
        "neighbourhood": "Neighbourhood or pincode",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "room_type": "Type of room",
        "price": "Price",
        "reviews_per_month": "Number of reviews per month",
        "calculated_host_listings_count": (
            "Total number of listings by the host"
        ),
        "availability_365": (
            "Availability of the property in the last year in days"
        ),
        "city": "City",
        "minimum_nights": "Minimum nights for a reservation",
        "number_of_reviews": "Total number of reviews",
    }
    df.rename(columns=col_name_to_desc, inplace=True)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  task_information = types.create_task_info(
      task_type="regression",
      target_key="Price",
      dataset_name="US AirBnB",
  )

  return df, task_information


def load_usa_housing() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed US Real Estate Listings dataset."""
  # Source:
  processed_data_path = BASE_PATHS["ingestables"] / "usa_housing.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing US Real Estate Listings dataset...")
    data, _ = read_arff(
        vertical_name="01_real_estate", dataset_name="02_usa_housing"
    )
    df = pd.DataFrame(data)

    # Drop redundant, useless, ID-type and missing features
    df.drop(
        columns=["id", "url", "region_url", "image_url"],
        inplace=True,
        errors="ignore",
    )

    boolean_feats = [k for k, v in df.nunique().to_dict().items() if v == 2]
    rep_str = {"0.0": "no", "1.0": "yes"}
    df.loc[:, boolean_feats] = (
        df.loc[:, boolean_feats].astype(str).replace(to_replace=rep_str)
    )

    # Change values
    rep_str = {
        "w/d in unit": "washer and dryer in unit",
        "w/d hookups": "washer and dryer hookups available",
        "laundry in bldg": "laundry in building",
        None: "Information unavailable",
    }
    df["laundry_options"] = df["laundry_options"].replace(to_replace=rep_str)
    df["parking_options"] = df["parking_options"].replace(
        to_replace={None: "Information unavailable"}
    )

    # Drop columns with missing latitude, longitude and description
    df.dropna(axis=0, inplace=True)

    rep_str = {
        "ca": "California",
        "co": "Colorado",
        "ct": "Connecticut",
        "dc": "District of Columbia",
        "fl": "Florida",
        "de": "Delaware",
        "ga": "Georgia",
        "hi": "Hawaii",
        "id": "Idaho",
        "il": "Illinois",
        "in": "Indiana",
        "ia": "Iowa",
        "ks": "Kansas",
        "ky": "Kentucky",
        "la": "Louisiana",
        "me": "Maine",
        "mi": "Michigan",
        "md": "Maryland",
        "ma": "Massachusetts",
        "mn": "Minnesota",
        "ms": "Mississippi",
        "nc": "North Carolina",
        "mo": "Missouri",
        "mt": "Montana",
        "ne": "Nebraska",
        "nv": "Nevada",
        "nj": "New Jersey",
        "nm": "New Mexico",
        "ny": "New York",
        "nh": "New Hampshire",
        "oh": "Ohio",
        "nd": "North Dakota",
        "ok": "Oklahoma",
        "or": "Oregon",
        "pa": "Pennsylvania",
        "ri": "Rhode Island",
        "sc": "South Carolina",
        "sd": "South Dakota",
        "tx": "Texas",
        "ut": "Utah",
        "va": "Virginia",
        "vt": "Vermont",
        "wa": "Washington",
        "wv": "West Virginia",
        "wi": "Wisconsin",
        "wy": "Wyoming",
        "al": "Alabama",
        "az": "Arizona",
        "ak": "Alaska",
        "ar": "Arkansas",
    }
    df["state"] = df["state"].replace(to_replace=rep_str)

    # Clean the description column
    df["description"] = df["description"].map(
        lambda x: preprocessing.clean_text(text=x, truncate_len=2500),
        na_action="ignore",
    )
    df.dropna(axis=0, inplace=True)  # Drop some rows with NaN descriptions

    # Refine column names
    col_name_to_desc = {
        "region": "Region",
        "price": "Price",
        "type": "Type of property",
        "sqfeet": "Area in square feet",
        "beds": "Number of beds",
        "baths": "Number of bathrooms",
        "cats_allowed": "Whether cats are allowed",
        "dogs_allowed": "Whether dogs are allowed",
        "smoking_allowed": "Whether smoking is allowed",
        "wheelchair_access": "Property is wheelchair accessible",
        "electric_vehicle_charge": (
            "Property has electric vehicle charging station"
        ),
        "comes_furnished": "Property is furnished",
        "laundry_options": "Laundry options",
        "parking_options": "Parking options",
        "description": "Description",
        "lat": "Latitude of property",
        "long": "Longitude of property",
        "state": "US State",
    }
    df.rename(columns=col_name_to_desc, inplace=True)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  task_information = types.create_task_info(
      task_type="regression",
      target_key="Price",
      dataset_name="USA Housing",
  )

  return df, task_information


def load_nashville_housing() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed Nashville Housing dataset."""
  # Source: https://www.kaggle.com/datasets/tmthyjames/nashville-housing-data
  processed_data_path = BASE_PATHS["ingestables"] / "nashville_housing.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing Nashvilled Housing dataset...")
    data, _ = read_arff(
        vertical_name="01_real_estate", dataset_name="04_nashville_housing"
    )
    df = pd.DataFrame(data)

    # Data Processing Rationale
    # 'Unnamed:_0' and 'Unnamed:_0.1' are indices --> Remove
    # Parcel ID, Suite/_Condo___#, Legal_Reference are likely to not be useful
    # Many features have almost 60% missing values --> Remove
    df.drop(
        columns=[
            "Unnamed:_0",
            "Unnamed:_0.1",
            "Parcel_ID",
            "Suite/_Condo___#",
            "Legal_Reference",
            "image",
            "Owner_Name",
            "State",  # Only one state
        ],
        inplace=True,
        errors="ignore",
    )
    # Many rows with almost 20 missing features
    df.dropna(axis=0, inplace=True)

    textual_columns = [
        "Land_Use",
        "Property_Address",
        "Property_City",
        "Address",
        "City",
        "Tax_District",
        "Exterior_Wall",
    ]
    df[textual_columns] = df[textual_columns].map(
        lambda x: preprocessing.clean_text(text=x, truncate_len=2500),
        na_action="ignore",
    )
    # Change values
    rep_str = {
        "greenbelt res grrenbelt res": "greenbelt (residential)",
        "vacant res land": "vacant residential land",
        "zero lot line": "zero lot line dwelling",
        "residential combo misc": "residential combo/miscellaneous",
        "office bldg one or two stories": (
            "office building (one or two stories)"
        ),
    }
    df["Land_Use"] = df["Land_Use"].replace(to_replace=rep_str)

    rep_str = {
        "PT BSMT": "Pier and Beam Basement",
        "SLAB": "Slab",
        "FULL BSMT": "Full Basement",
        "CRAWL": "Crawl Space",
        "PIERS": "Piers",
        "TYPICAL": "Typical",
    }
    df["Foundation_Type"] = df["Foundation_Type"].replace(to_replace=rep_str)

    rep_str = {
        "conc blk": "concrete block",
        "precast conc": "precast concrete",
    }
    df["Exterior_Wall"] = df["Exterior_Wall"].replace(to_replace=rep_str)
    df["Grade"] = df["Grade"].map(lambda x: x.strip())

    df.columns = [" ".join(i.split("_")) for i in list(df.columns)]

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  task_information = types.create_task_info(
      task_type="regression",
      target_key="Sale Price",
      dataset_name="Nashville Housing",
  )

  return df, task_information


def load_us_real_estate() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed US Real Estate Listings by Zip Code dataset."""
  # Source: https://www.openml.org/search?type=data&status=any&sort=qualities.NumberOfInstances&id=43631  pylint: disable=line-too-long

  processed_data_path = BASE_PATHS["ingestables"] / "us_real_estate.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing US Real Estate Listings by Zip Code dataset...")

    data, _ = read_arff(
        vertical_name="01_real_estate", dataset_name="01_us_real_estate"
    )
    df = pd.DataFrame(data)

    # Drop columns with many missing values
    df.drop(
        columns=[
            "Footnote",
            "Price_Increase_Count_M/M",
            "Price_Increase_Count_Y/Y",
            "Pending_Listing_Count_M/M",
            "Pending_Listing_Count_Y/Y",
            "Median_Listing_Price_Y/Y",
            "Active_Listing_Count_Y/Y",
            "Days_on_Market_Y/Y",
            "New_Listing_Count_Y/Y",
            "Price_Decrease_Count_Y/Y",
        ],
        inplace=True,
        errors="ignore",
    )
    df.dropna(axis=0, inplace=True)  # Drop remaining rows with missing values

    # Format column name
    col_name_to_desc = {
        "ZipCode": "Zip code",
        "ZipName": "City, State",
        "Median_Listing_Price": (
            "Median listing price within specified geography and month"
        ),
        "Median_Listing_Price_M/M": (
            "Month on month change in median listing price"
        ),
        "Active_Listing_Count_": (
            "Number of active listings within specified geography and month"
        ),
        "Active_Listing_Count_M/M": "Month on month change in active listings",
        "Days_on_Market_": "Number of days marks",
        "Days_on_Market_M/M": "Month on month change in number of days market ",
        "New_Listing_Count_": (
            "Number of new listings added to the market within specified"
            + " geography"
        ),
        "New_Listing_Count_M/M": (
            "Month on month change in number of new listings added to the"
            " market"
        ),
        "Price_Increase_Count_": (
            "Number of listings which have had their price increased within"
            + " specified geography"
        ),
        "Price_Decrease_Count_": (
            "Number of listings which have had their price decreased within"
            + " specified geography"
        ),
        "Price_Decrease_Count_M/M": (
            "Change in number of listings which have had their price decreased"
        ),
        "Pending_Listing_Count_": (
            "Number of pending listings within specified geography and month"
        ),
        "Avg_Listing_Price": (
            "Average listing price within specified geography and month"
        ),
        "Avg_Listing_Price_M/M": (
            "Month on month change in average listing price"
        ),
        "Avg_Listing_Price_Y/Y": "Year on year change in average listing price",
        "Total_Listing_Count": (
            "Total number of listings within specified geography and month"
        ),
        "Total_Listing_Count_M/M": (
            "Month on month change in total number of listings"
        ),
        "Total_Listing_Count_Y/Y": (
            "Year on year change in total number of listings"
        ),
        "Pending_Ratio": "Pending ratio within specified geography and month",
        "Pending_Ratio_M/M": "Month on month change in pending ratio",
        "Pending_Ratio_Y/Y": "Year on year change in pending ratio",
    }
    df.rename(columns=col_name_to_desc, inplace=True)

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  # target_key = "Median listing price within specified geography and month"
  target_key = "Average listing price within specified geography and month"
  dataset_name = "US Real Estate Listings by Zip Code"
  task_information = types.create_task_info(
      task_type="regression",
      target_key=target_key,
      dataset_name=dataset_name,
  )

  return df, task_information


def load_nyc_housing() -> Tuple[pd.DataFrame, types.TaskInfo]:
  """Loads the pre-proprocessed NYC Housing Data 2003 -- 2019 dataset."""
  # Source: https://www.openml.org/search?type=data&status=any&sort=qualities.NumberOfInstances&id=43633  pylint: disable=line-too-long

  processed_data_path = BASE_PATHS["ingestables"] / "nyc_housing.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
  else:
    logging.info("Preprocessing NYC Housing Data 2003 -- 2019 dataset...")

    data, _ = read_arff(
        vertical_name="01_real_estate", dataset_name="00_nyc_housing"
    )
    df = pd.DataFrame(data)

    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(inplace=True)  # Some rows are duplicate
    df.reset_index(inplace=True, drop=True)

    # Drop rows with abnormal values
    abnormal_sale_prices = np.where(df["SALE_PRICE"] < 1000)[0]
    abnormal_zip_codes = np.where(df["ZIP_CODE"] == 0)[0]
    abnormal_years = np.where(df["YEAR_BUILT"] <= 1800.0)[0]
    abnormal_land_sq_feet = np.where(df["LAND_SQUARE_FEET"] <= 50.0)[0]
    abnormal_gross_sq_feet = np.where(df["GROSS_SQUARE_FEET"] <= 50.0)[0]
    abnormal_values = np.concatenate([
        abnormal_zip_codes,
        abnormal_sale_prices,
        abnormal_years,
        abnormal_land_sq_feet,
        abnormal_gross_sq_feet,
    ])
    abnormal_values = sorted(np.unique(abnormal_values))
    df.drop(index=abnormal_values, inplace=True, errors="ignore")

    # Format values
    str_cols = ["NEIGHBORHOOD", "BUILDING_CLASS_CATEGORY", "ADDRESS"]
    df.loc[:, str_cols] = df.loc[:, str_cols].map(
        lambda x: preprocessing.clean_text(text=x, truncate_len=128),
        na_action="ignore",
    )
    df.loc[:, "SALE_DATE"] = df.loc[:, "SALE_DATE"].map(lambda x: x[:-9])

    repr_str = {
        1.0: "Manhattan, New York",
        2.0: "Bronx, New York",
        3.0: "Brooklyn, New York",
        4.0: "Queens, New York",
        5.0: "Staten Island, New York",
    }
    df["BOROUGH"] = df["BOROUGH"].replace(to_replace=repr_str)
    df.columns = [i.replace("_", " ").title() for i in list(df.columns)]

    # Save pre-processed datast
    with processed_data_path.open("w") as f:
      df.to_csv(f, index=False)

  task_info = types.create_task_info(
      task_type="regression",
      target_key="Sale Price",
      dataset_name="NYC Housing",
  )
  return df, task_info


ingestables_dataset_name_to_class = {
    "autos": load_autos,
    "home_credit": load_home_credit,
    "give_me_some_credit": load_give_me_some_credit,
    "south_africa_debt": load_south_africa_debt,
    "indonesian_telecom_delinquency": load_indonesian_telecom_delinquency,
    "nyc_housing": load_nyc_housing,
    "us_real_estate": load_us_real_estate,
    "usa_housing": load_usa_housing,
    "us_airbnb": load_us_airbnb,
    "nashville_housing": load_nashville_housing,
}
