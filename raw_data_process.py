import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from collections import Counter

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


# function to transfer the csv to parquet
def raw_flow_to_parquet(gz_flow_dir: str, parquet_flow_dir: str) -> str:
    """
    This function converts downloaded 5-min flow gz files in gz_flow_dir
    to parquet formats and save it in parquet_flow_dir
    """
    all_raw_flow_data = os.listdir(gz_flow_dir)
    gz_files = [file for file in os.listdir(gz_flow_dir) if file.endswith(".gz")]

    # loop through all gz lfow data

    for gz_flow_data in gz_files:
        # import raw flow data
        raw_flow_df = pd.read_csv(
            os.path.join(RAW_FLOW_DIRECTORY, gz_flow_data),
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
        parquet_prefix = gz_flow_data[:3]
        parquet_year = gz_flow_data.split("_")[4]
        parquet_month = gz_flow_data.split("_")[5]
        parquet_day = gz_flow_data.split("_")[6][:2]
        parquet_new_name = (
            parquet_prefix
            + "_5min_flow_"
            + parquet_year
            + parquet_month
            + parquet_day
            + ".parquet"
        )
        # save df to parquet
        raw_flow_df.to_parquet(os.path.join(parquet_flow_dir, parquet_new_name))

    return "All gz files converted to parquet."
