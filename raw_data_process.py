import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RAW_DATA_PATH = 'raw_data/d12_text_station_5min_2019_10_08.txt.gz'
COLS_TO_IMPORT = list(range(0, 12))
df = pd.read_csv(RAW_DATA_PATH, compression='gzip', delimiter=',', header=None, usecols=COLS_TO_IMPORT)
cols_name = ['timestamp', 'station', 'district', 'freeway_num', 'direction', \
     'lane_type', 'station_length', 'samples', 'pct_observed', 'total_flow', \
     'avg_occupancy', 'avg_speed']
df.columns = cols_name

