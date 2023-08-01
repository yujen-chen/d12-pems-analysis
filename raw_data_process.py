import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from collections import Counter
from pyproj import Transformer
import matplotlib.pyplot as plt

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


def load_files(data_dir: str):
    # raw_flow_data =
    pass


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


"""
Load the station meta data
"""


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

    # rename columns
    pm_cols_name = [
        "freeway_num",
        "district",
        "abs_pm",
        "aligncode",
        "latitude",
        "longitude",
    ]
    raw_pm_df.columns = pm_cols_name

    # save it to parquet
    parquet_pm_name = "raw_pm.parquet"
    raw_pm_df.to_parquet(os.path.join(parquet_pm_dir, parquet_pm_name))

    return raw_pm_df
