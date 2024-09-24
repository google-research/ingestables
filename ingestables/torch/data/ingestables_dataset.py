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

"""Functions to pre-process and load ingestables datasets."""

from etils import epath
from ingestables.torch.data import data_utils
import numpy as np
import pandas as pd

ROOT_PATH = epath.Path("~/ingestables")
BASE_PATH = ROOT_PATH / "/datasets/verticals/processed"


def load_autos() -> pd.DataFrame:
  """Loads the pre-proprocessed Autos dataset."""
  # Source: https://www.kaggle.com/datasets/toramky/automobile-dataset

  processed_data_path = BASE_PATH / "autos.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
      vertical_name="00_risk", dataset_name="00_autos"
  )

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

  return df


def load_home_credit() -> pd.DataFrame:
  """Loads the pre-proprocessed Home Credit dataset."""
  # Source: https://www.kaggle.com/competitions/home-credit-default-risk/data

  processed_data_path = BASE_PATH / "home_credit.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
      vertical_name="00_risk", dataset_name="02_home_credit"
  )

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
  col_desc_path = ROOT_PATH / "/datasets/verticals/00_risk/02_home_credit/column_description_cleaned.csv"
  with col_desc_path.open("r") as f:
    col_desc = pd.read_csv(f)

  col_name_to_desc = col_desc.set_index("Feature Name").to_dict()[
      "Feature Description"
  ]
  df.rename(columns=col_name_to_desc, inplace=True)
  df.rename(columns={"Class": "Consumer repays the loan"}, inplace=True)

  # Save pre-processed datast
  with processed_data_path.open("w") as f:
    df.to_csv(f, index=False)

  return df


def load_give_me_some_credit() -> pd.DataFrame:
  """Loads the pre-proprocessed Give Me Some Credit dataset."""
  # Source: https://www.kaggle.com/competitions/GiveMeSomeCredit/data

  processed_data_path = BASE_PATH / "give_me_some_credit.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
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
          "Monthly debt payments, alimony, living costs divided by monthy gross"
          + " income"
      ),
      "MonthlyIncome": "Monthly income",
      "NumberOfOpenCreditLinesAndLoans": (
          "Number of Open loans (e.g. car loan or mortgage) and Lines of credit"
          + " (e.g. credit cards)"
      ),
      "NumberOfTimes90DaysLate": (
          "Number of times borrower has been 90 days or more past due."
      ),
      "NumberRealEstateLoansOrLines": (
          "Number of mortgage and real estate loans including home equity lines"
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

  return df


def load_south_africa_debt() -> pd.DataFrame:
  """Loads the pre-proprocessed Municipal Debt Risk Analysis dataset."""
  # Source: https://www.kaggle.com/datasets/dmsconsultingsa/municipal-debt-risk-analysis  pylint: disable=line-too-long

  processed_data_path = BASE_PATH / "south_africa_debt.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
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

  return df


def load_indonesian_telecom_delinquency() -> pd.DataFrame:
  """Loads the pre-proprocessed Indonesian Telecom Delinquency dataset."""
  # Source: https://www.kaggle.com/datasets/dmsconsultingsa/municipal-debt-risk-analysis  pylint: disable=line-too-long
  # Resources: https://github.com/thamizhdatatrained/Micro-Credit-Loan-Defaulter-Project/tree/main  pylint: disable=line-too-long

  processed_data_path = BASE_PATH / "indonesian_telecom_delinquency.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  path = ROOT_PATH / "datasets/verticals/00_risk/05_indonesian_telecom_delinquency/raw_dataset.csv"
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
          "Daily amount spent from main account, averaged over last 30 days (in"
          + " Indonesian Rupiah)"
      ),
      "daily_decr90": (
          "Daily amount spent from main account, averaged over last 90 days (in"
          + " Indonesian Rupiah)"
      ),
      "rental30": "Average main account balance over last 30 days",
      "rental90": "Average main account balance over last 90 days",
      "last_rech_date_ma": "Number of days till last recharge of main account",
      "last_rech_date_da": "Number of days till last recharge of data account",
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
          "Median of amount of recharges done in main account over last 30 days"
          + " at user level (in Indonesian Rupiah)"
      ),
      "medianmarechprebal30": (
          "Median of main account balance just before recharge in last 30 days"
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
          "Median of amount of recharges done in main account over last 90 days"
          + " at user level (in Indonesian Rupiah)"
      ),
      "medianmarechprebal90": (
          "Median of main account balance just before recharge in last 90 days"
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

  return df


def load_us_airbnb() -> pd.DataFrame:
  """Loads the pre-proprocessed US AirBnB dataset."""
  # Source: https://www.kaggle.com/datasets/kritikseth/us-airbnb-open-data

  processed_data_path = BASE_PATH / "us_airbnb.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
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
  df["reviews_per_month"] = df["reviews_per_month"].replace(to_replace=repr_str)
  repr_str = {"Entire home/apt": "Entire home or apartment"}
  df["room_type"] = df["room_type"].replace(to_replace=repr_str)
  # Clean the name column
  df["name"] = df["name"].map(
      lambda x: data_utils.clean_text(text=x, truncate_len=128)
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
      "calculated_host_listings_count": "Total number of listings by the host",
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

  return df


def load_usa_housing() -> pd.DataFrame:
  """Loads the pre-proprocessed US Real Estate Listings dataset."""
  # Source:
  processed_data_path = BASE_PATH / "usa_housing.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
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
      lambda x: data_utils.clean_text(text=x, truncate_len=2500),
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

  return df


def load_us_real_estate() -> pd.DataFrame:
  """Loads the pre-proprocessed US Real Estate Listings by Zip Code dataset."""
  # Source: https://www.openml.org/search?type=data&status=any&sort=qualities.NumberOfInstances&id=43631  pylint: disable=line-too-long

  processed_data_path = BASE_PATH / "us_real_estate.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
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
          "Month on month change in number of new listings added to the market"
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
      "Avg_Listing_Price_M/M": "Month on month change in average listing price",
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

  return df


def load_nyc_housing() -> pd.DataFrame:
  """Loads the pre-proprocessed NYC Housing Data 2003 -- 2019 dataset."""
  # Source: https://www.openml.org/search?type=data&status=any&sort=qualities.NumberOfInstances&id=43633  pylint: disable=line-too-long

  processed_data_path = BASE_PATH / "nyc_housing.csv"

  if processed_data_path.exists():
    with processed_data_path.open("r") as f:
      df = pd.read_csv(f)
      return df

  data, _ = data_utils.read_arff(
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
      lambda x: data_utils.clean_text(text=x, truncate_len=128),
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

  return df
