import os
from functools import partial

import pandas as pd
import statsforecast
from statsforecast import StatsForecast
from statsforecast.feature_engineering import mstl_decomposition
from statsforecast.models import ARIMA, MSTL
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import smape, mase

# this makes it so that the outputs of the predict methods have the id as a column 
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'
df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
uids = df['unique_id'].unique()[:10]
df = df[df['unique_id'].isin(uids)]
df.head()
