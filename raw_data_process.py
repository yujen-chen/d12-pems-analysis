import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from collections import Counter
from pyproj import Transformer
import matplotlib.pyplot as plt
import re

# DIRECTORY PATH
RAW_FLOW_DIRECTORY = "raw_data/flow"
PARQUET_FLOW_DIRECTORY = "data/flow"

RAW_FLOW_COLS_TO_IMPORT = list(range(0, 12))

RAW_FLOW_COLS_NAMES = [
    "timestamp",
    "station",
    "district",
    "freeway_num",
    "direction",
    "lane_type",
    "station_length",
    "samples",
    "pct_observed",
    "total_flow",
    "avg_occupancy",
    "avg_speed",
]

"""
# TODO:

"""


"""
Load Parquet Data
"""


def load_parquet_data(data_dir_path: str):
    # define all the folders
    parquet_flow_dir = os.path.join(data_dir_path, "flow")
    parquet_station_dir = os.path.join(data_dir_path, "station_meta")
    parquet_pm_dir = os.path.join(data_dir_path, "pm")
    parquet_flow_data = os.listdir(parquet_flow_dir)
    parquet_station_data = os.listdir(parquet_station_dir)
    parquet_pm_data = os.listdir(parquet_pm_dir)

    # station_df
    station_df = pd.read_parquet(
        os.path.join(parquet_station_dir, parquet_station_data[0])
    )

    # pm_df
    pm_df = pd.read_parquet(os.path.join(parquet_pm_dir, parquet_pm_data[0]))

    # flow dict
    raw_flow_dfs = {}
    for flow_parquet in parquet_flow_data:
        parquet_flow_key = (
            flow_parquet[:3] + "_" + re.search(r"\d{8}", flow_parquet).group()
        )
        raw_flow_dfs[parquet_flow_key] = pd.read_parquet(
            os.path.join(parquet_flow_dir, flow_parquet)
        )

        assert (
            pd.Series(raw_flow_dfs[parquet_flow_key]["station"].unique())
            .isin(station_df["station"])
            .all()
        ), "station list mismatch"

        raw_flow_dfs[parquet_flow_key] = pd.merge(
            raw_flow_dfs[parquet_flow_key],
            station_df[["station", "abs_pm"]],
            on="station",
            how="left",
        )

        assert (raw_flow_dfs[parquet_flow_key]["station"].isnull().sum() == 0) & (
            raw_flow_dfs[parquet_flow_key]["abs_pm"].isnull().sum() == 0
        ), "station and/or abs-pm mismatch"

    return raw_flow_dfs, station_df, pm_df


"""
Load csv raw data and save as parquet
"""


# function to transfer the csv to parquet
def raw_flow_to_parquet(gz_flow_dir: str, parquet_flow_dir: str) -> dict:
    """
    This function converts downloaded 5-min flow gz files in gz_flow_dir
    to parquet formats, save it in parquet_flow_dir, and return a dictionary of raw flow df
    """
    all_raw_flow_data = os.listdir(gz_flow_dir)
    gz_files = [file for file in os.listdir(gz_flow_dir) if file.endswith(".gz")]

    # Create an empty dictionary to store the DataFrames
    raw_flow_dfs = {}

    # loop through all gz flow data
    for gz_flow_data in gz_files:
        # config the new flow_df name
        raw_flow_district = gz_flow_data[:3]
        raw_flow_year = gz_flow_data.split("_")[4]
        raw_flow_month = gz_flow_data.split("_")[5]
        raw_flow_day = gz_flow_data.split("_")[6][:2]
        raw_flow_dictionary_key = (
            raw_flow_district + "_" + raw_flow_year + raw_flow_month + raw_flow_day
        )

        # import raw flow data
        raw_flow_df = pd.read_csv(
            os.path.join(gz_flow_dir, gz_flow_data),
            compression="gzip",
            delimiter=",",
            header=None,
            usecols=RAW_FLOW_COLS_TO_IMPORT,
        )

        # rename the columns
        raw_flow_df.columns = RAW_FLOW_COLS_NAMES

        # convert timestamp to datetime format
        raw_flow_df["timestamp"] = pd.to_datetime(
            raw_flow_df["timestamp"]
        ).dt.tz_localize(None)
        raw_flow_df["timestamp"] = raw_flow_df["timestamp"].dt.tz_localize(
            "America/Los_Angeles"
        )

        # check timestamp hour and minutes
        unique_timestamps = raw_flow_df["timestamp"].unique()
        unique_hours = unique_timestamps.hour
        unique_minutes = unique_timestamps.minute
        assert (
            (len(unique_timestamps) == 288)
            & (set(unique_hours) == set(range(0, 24)))
            & (set(unique_minutes) == set(range(0, 60, 5)))
        ), "please check timestamp column"

        hour_counts = Counter(unique_hours)
        assert all(
            hour_counts[hour] == 12 for hour in range(24)
        ), "hour is wrong, check timestamp column"

        minute_counts = Counter(unique_minutes)
        assert all(
            minute_counts[minute] == 24 for minute in range(0, 60, 5)
        ), "minute is wrong, check timestamp column"

        # keep only ML and HV for lane_type
        raw_flow_df = raw_flow_df[raw_flow_df["lane_type"].isin(["ML", "HV"])]

        # config parquet file name
        parquet_new_name = (
            raw_flow_district
            + "_5min_flow_"
            + raw_flow_year
            + raw_flow_month
            + raw_flow_day
            + ".parquet"
        )

        # save df to parquet
        raw_flow_df.to_parquet(os.path.join(parquet_flow_dir, parquet_new_name))

        # save it to dictionary
        raw_flow_dfs[raw_flow_dictionary_key] = raw_flow_df

    return raw_flow_dfs


def tweak_station_meta(
    raw_station_dir: str, year: int, raw_flow_df: pd.DataFrame, parquet_station_dir: str
) -> pd.DataFrame:
    """
    This function reads the station meta data for a given year, returns a DataFrame
    with renamed columns and saves it as parquet

    Parameters:
    raw_station_dir (str): The directory where the raw station meta files are stored.
    year (int): The year for which the station meta data is to be read.
    parquet_station_dir (str): The directory where the parquet station meta files are stored.

    Returns:
    pd.DataFrame: The station meta data for the given year with renamed columns.
    """

    # locate the directory and file that matches the
    all_station_metas = os.listdir(raw_station_dir)
    station_meta_year = [file for file in all_station_metas if str(year) in file]

    # Check if the file for the year exists
    if not station_meta_year:
        raise FileNotFoundError(
            f"No file found for the year {year} in the directory {raw_station_dir}"
        )

    # read station meta file
    raw_station_cols = [
        "ID",
        "Fwy",
        "Dir",
        "District",
        "County",
        "Abs_PM",
        "Latitude",
        "Longitude",
    ]
    raw_station_df = pd.read_csv(
        os.path.join(raw_station_dir, station_meta_year[0]),
        delimiter="\t",
        usecols=raw_station_cols,
    )

    # round abs_pm to two decimal places
    raw_station_df["Abs_PM"] = raw_station_df["Abs_PM"].round(2)

    sdf_col_names = [
        "station",
        "freeway_num",
        "direction",
        "district",
        "county",
        "abs_pm",
        "latitude",
        "longitude",
    ]
    raw_station_df.columns = sdf_col_names

    # check if the stations in raw_flow_df are all included in raw_station_df
    if not raw_flow_df["station"].isin(raw_station_df["station"]).all():
        raise ValueError("Some stations in flow data are not in the station meta data")

    parquet_station_meta = "d12_station_" + str(year) + ".parquet"
    raw_station_df.to_parquet(os.path.join(parquet_station_dir, parquet_station_meta))

    return raw_station_df


# tweak pm
def tweak_pm_df(raw_pm_dir: str, parquet_pm_dir: str) -> pd.DataFrame:
    # locate raw pm data
    raw_pm_data = os.listdir(raw_pm_dir)
    if not raw_pm_data:
        raise ValueError("The directory is empty")
    # columns to be imported
    raw_pmcols_to_import = [0, 1, 3, 8, 10, 16]

    raw_pm_df = pd.read_csv(
        os.path.join(raw_pm_dir, raw_pm_data[0]), usecols=raw_pmcols_to_import
    )

    # # convert coordinates
    # in_proj = Proj("epsg:3857")
    # out_proj = Proj("epsg:4326")

    x = raw_pm_df["X"]  # Original x-coordinate
    y = raw_pm_df["Y"]  # Original y-coordinate

    # Input coordinate system (Web Mercator) EPSG:3857
    # Output coordinate system (WGS84) EPSG:4326
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
    raw_pm_df["latitude"], raw_pm_df["longitude"] = transformer.transform(x, y)
    # pmdf["longitude"], pmdf["latitude"] = transform(in_proj, out_proj, x, y)

    # drop original X and Y
    raw_pm_df = raw_pm_df.drop(["X", "Y"], axis=1)

    # Keep AlignCode only for left and drop AlignCode
    raw_pm_df = raw_pm_df[raw_pm_df["AlignCode"] == "Left"].drop(
        columns=["AlignCode"], axis=1
    )

    # Round PM column to two decimal place
    raw_pm_df["PM"] = raw_pm_df["PM"].round(2)

    # rename columns
    pm_cols_name = [
        "freeway_num",
        "district",
        "abs_pm",
        "latitude",
        "longitude",
    ]
    raw_pm_df.columns = pm_cols_name

    # save it to parquet
    parquet_pm_name = "raw_pm.parquet"
    raw_pm_df.to_parquet(os.path.join(parquet_pm_dir, parquet_pm_name))

    return raw_pm_df


"""
Flow data handling

- Select the route
- Assign NaN values and merge with station df to get abs-pm info
-
"""


# select route from raw_flow_dict
def select_route(
    raw_flow_df: pd.DataFrame,
    freeway_num: int,
    direction: str,
    lane_type: str,
) -> pd.DataFrame:
    """
    This function extracts raw flow data for input route from raw flow dictionary
    and save it as dictionary
    """

    required_columns_before = {"freeway_num", "direction", "lane_type"}

    required_columns_after = {
        "timestamp",
        "station",
        "district",
        "freeway_num",
        "direction",
        "lane_type",
        "station_length",
        "samples",
        "pct_observed",
        "total_flow",
        "avg_occupancy",
        "avg_speed",
    }

    # copy the input data
    df = raw_flow_df.copy()

    # use assert to check the required columns
    assert required_columns_before.issubset(
        df.columns
    ), f"{key} is missing some required columns before processing"

    # filter the df based on route number, direction, and lane type
    df = df.query(
        "freeway_num == @freeway_num & direction == @direction & lane_type == @lane_type"
    )

    # sort by timestamp then reset index
    df = df.sort_values("timestamp").reset_index(drop=True)

    assert df.shape[1] == 13, "The columns of route df is wrong"

    assert required_columns_after.issubset(
        df.columns
    ), f"Some columns are not included in {key}, check raw data"

    return df


# assign null pivot and merge with sdf
def process_flow(route_flow_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function does the following:
    - Assign NaN to flow if the 'pct_observed' is less than 70
    - Pivot the raw df with index=[station, abs_pm], columns=timestamp, values=total_flow
    - Drop the row if the row includes more than 50% of NaN
    """

    # backup input data
    df = route_flow_df.copy()
    # sdf_copy = sdf[["station", "abs_pm"]].copy()

    # rename the columns for pivot df
    pivot_col_names = [
        f"{hour:02d}{minute:02d}" for hour in range(0, 24) for minute in range(0, 60, 5)
    ]
    pivot_col_names = ["station", "abs_pm"] + pivot_col_names

    # required columns before process
    required_columns_before = {
        "pct_observed",
        "total_flow",
        "timestamp",
        "station",
        "abs_pm",
    }

    # check the columns
    assert required_columns_before.issubset(
        df.columns
    ), f"{key} is missing some required columns."
    assert (
        df["freeway_num"].nunique() == 1
    ), f"{key} should have only one unique value in the 'station' column."
    assert (
        df["direction"].nunique() == 1
    ), f"{key} should have only one unique value in the 'station' column."
    assert (
        df["lane_type"].nunique() == 1
    ), f"{key} should have only one unique value in the 'station' column."

    # filter flow_df and assign NaN
    pct_observed_mask = df["pct_observed"] < 70
    df.loc[pct_observed_mask, "total_flow"] = np.nan

    # pivot df, each row is a station and each column is 5-min interval
    _p_df = df.pivot(
        index=["station", "abs_pm"], columns="timestamp", values="total_flow"
    ).reset_index()

    # if the whole row are NaN values, drop it
    drop_raw_threshold = len(df.columns) * 0
    # drop the raw if the row has at least three non-NaN values (two for station, abs_pm, and one for flow)
    _p_df = _p_df.dropna(thresh=3)

    # rename the columns
    _p_df.columns = pivot_col_names

    # sort by abs_pm then reset index
    _p_df = _p_df.sort_values("abs_pm").reset_index(drop=True)

    return _p_df


# Merge with pm and Interpolation


def flow_interpolation(
    flow_df: pd.DataFrame, pmdf: pd.DataFrame, district: int, route: int
):
    _df = flow_df.copy()
    # filter pmdf
    route_pm = (
        pmdf.query("district==@district & freeway_num==@route")
        .sort_values("abs_pm")
        .reset_index(drop=True)
    )

    # create a list of bins that will be used for PM range later
    bins = [0] + route_pm["abs_pm"].tolist() + [np.inf]

    # append the bins to route_merged_df
    _df = _df.sort_values("abs_pm").reset_index(drop=True)
    _df["abs_pm_range"] = pd.cut(_df["abs_pm"], bins=bins)

    # calculate mean flow
    time_col_names = [
        f"{hour:02d}{minute:02d}" for hour in range(0, 24) for minute in range(0, 60, 5)
    ]
    col_names = ["abs_pm_range"] + time_col_names
    _mean_flow_df = _df[col_names].groupby("abs_pm_range").mean().reset_index(drop=True)

    # delete the last row (last_pm, np.inf]
    _mean_flow_df = _mean_flow_df.drop(_mean_flow_df.index[-1])

    # check if rows of route_mean_flow and pm_df are the same
    if len(_mean_flow_df) == len(route_pm):
        _res_df = route_pm.merge(_mean_flow_df, left_index=True, right_index=True)

        # forward fillna for pm within route_flow_range, back fillna beyound route_flow_range
        _res_df = _res_df.fillna(method="bfill", axis=0).fillna(method="ffill", axis=0)

        # save non-interpolation df
        _res_non_int_df = _res_df.copy()

        # interpolate nan across columns (across time)
        _res_df[time_col_names] = _res_df[time_col_names].interpolate(axis=1).round(0)
        return _res_non_int_df, _res_df
    else:
        return "Error, check input dfs"
