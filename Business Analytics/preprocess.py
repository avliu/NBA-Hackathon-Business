import pandas as pd
from fastai.tabular import add_datepart
import re


def normalize(df, columns):
    df.loc[:,columns] = \
        (df.loc[:,columns] -
         df.loc[:,columns].mean()) / \
        (df.loc[:,columns].max() -
         df.loc[:,columns].min())

# 1. add dateparts

# df_stats = pd.read_csv('training_set_stats.csv')
#
# df_stats_type = pd.get_dummies(df_stats.loc[:, 'Type'])
# df_stats_type = pd.concat((df_stats, df_stats_type), axis=1).drop(['Type'], axis=1)
#
# df_stats_datetime = df_stats_type
# add_datepart(df=df_stats_datetime, field_name='Created', time=True)
# df_stats_datetime.to_csv('training_set_datetimes.csv')

# df_stats = pd.read_csv('holdout_set_stats.csv')
#
# df_stats_type = pd.get_dummies(df_stats.loc[:, 'Type'])
# df_stats_type = pd.concat((df_stats, df_stats_type), axis=1).drop(['Type'], axis=1)
#
# df_stats_datetime = df_stats_type
# add_datepart(df=df_stats_datetime, field_name='Created', time=True)
# df_stats_datetime.to_csv('holdout_set_datetimes.csv')


# 2. drop irrelevant columns, consolidate hours and minutes, create training_set_final_0.csv

# df_stats_datetime = pd.read_csv('training_set_datetimes.csv')
# df_stats_datetime = df_stats_datetime.drop(columns=['Unnamed: 0', 'CreatedMonth', 'CreatedDay', 'CreatedWeek',
#                                                     'CreatedIs_month_end', 'CreatedIs_month_start',
#                                                     'CreatedIs_quarter_end', 'CreatedIs_quarter_start',
#                                                     'CreatedIs_year_start', 'CreatedIs_year_end',
#                                                     'CreatedSecond', 'CreatedElapsed', 'CreatedYear'])
# df_stats_datetime['CreatedTotalMinutes'] = df_stats_datetime.loc[:, 'CreatedHour']*60 +\
#                                            df_stats_datetime.loc[:,'CreatedMinute']
# df_stats_datetime = df_stats_datetime.drop(columns=['CreatedHour', 'CreatedMinute'])
# # df_stats_datetime['CreatedWeekend'] = (df_stats_datetime.loc[:,'CreatedDayofweek']).apply(lambda x: (1 if x > 3 else 0))
# # df_stats_datetime = df_stats_datetime.drop(columns=['CreatedDayofweek'])
#
# df_stats_normalized = df_stats_datetime
# normalize(df_stats_normalized, ('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes'))
#
# df_stats_normalized.to_csv('training_set_final_0.csv', index=False)


# df_stats_datetime = pd.read_csv('holdout_set_datetimes.csv')
# df_stats_datetime = df_stats_datetime.drop(columns=['Unnamed: 0', 'CreatedMonth', 'CreatedDay', 'CreatedWeek',
#                                                     'CreatedIs_month_end', 'CreatedIs_month_start',
#                                                     'CreatedIs_quarter_end', 'CreatedIs_quarter_start',
#                                                     'CreatedIs_year_start', 'CreatedIs_year_end',
#                                                     'CreatedSecond', 'CreatedElapsed', 'CreatedYear'])
# df_stats_datetime['CreatedTotalMinutes'] = df_stats_datetime.loc[:, 'CreatedHour']*60 +\
#                                            df_stats_datetime.loc[:,'CreatedMinute']
# df_stats_datetime = df_stats_datetime.drop(columns=['CreatedHour', 'CreatedMinute'])
# # df_stats_datetime['CreatedWeekend'] = (df_stats_datetime.loc[:,'CreatedDayofweek']).apply(lambda x: (1 if x > 3 else 0))
# # df_stats_datetime = df_stats_datetime.drop(columns=['CreatedDayofweek'])
#
# df_stats_normalized = df_stats_datetime
# df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')] = \
#     (df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')] -
#      df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')].mean()) / \
#     (df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')].max() -
#      df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')].min())
#
# df_stats_normalized.to_csv('holdout_set_final_0.csv', index=False)


# 3. add text-related stats to the new dataset: holdout_set_final_1.csv, training_set_stats_final_1.csv

# datasets = ('training_set', 'holdout_set')
#
# for dataset in datasets:
#     df_text = pd.read_csv(f'text_processing/{dataset}_same_stats_final.csv', index_col=0)
#     df_stats = pd.read_csv(f'{dataset}_final_0.csv')
#     df_all = df_stats.copy()
#
#     for c in df_text.columns:
#         df_all[c] = df_text.loc[:,c]
#
#     normalize(df_all, ('player_followers',  'team_followers', 'celebrity_followers', 'organization_followers'))
#
#     # hacky... should fix
#     df_all.loc[:, 'CreatedDayofweek'] = df_all.loc[:, 'CreatedDayofweek'] / 7
#     df_all.loc[:, 'player'] = df_all.loc[:, 'player'] / 9
#     df_all.loc[:, 'team'] = df_all.loc[:, 'team'] / 7
#     df_all.loc[:, 'celebrity'] = df_all.loc[:, 'celebrity'] / 4
#     df_all.loc[:, 'organization'] = df_all.loc[:, 'organization'] / 5
#
#     df_all.to_csv(f'{dataset}_final_1.csv', index=False)
