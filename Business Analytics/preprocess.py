import pandas as pd
from fastai.tabular import add_datepart


df_stats = pd.read_csv('training_set_stats.csv')

df_stats_type = pd.get_dummies(df_stats.loc[:, 'Type'])
df_stats_type = pd.concat((df_stats, df_stats_type), axis=1).drop(['Type'], axis=1)

df_stats_datetime = df_stats_type
add_datepart(df=df_stats_datetime, field_name='Created', time=True)
