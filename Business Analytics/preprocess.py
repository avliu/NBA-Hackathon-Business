import pandas as pd
from fastai.tabular import add_datepart
import re


# 1. add dateparts

# df_stats = pd.read_csv('training_set_stats.csv')
#
# df_stats_type = pd.get_dummies(df_stats.loc[:, 'Type'])
# df_stats_type = pd.concat((df_stats, df_stats_type), axis=1).drop(['Type'], axis=1)
#
# df_stats_datetime = df_stats_type
# add_datepart(df=df_stats_datetime, field_name='Created', time=True)
# df_stats_datetime.to_csv('training_set_datetimes.csv')


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
# df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')] = \
#     (df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')] -
#      df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')].mean()) / \
#     (df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')].max() -
#      df_stats_normalized.loc[:,('Followers at Posting','CreatedDayofyear','CreatedTotalMinutes')].min())
#
# df_stats_normalized.to_csv('training_set_final_0.csv', index=False)


# 3. prepare text data- map scraped instagram usernames to actual usernames
#    WRONG USERNAMES (changed only in training_set_same.csv and holdout_set_same.csv)
#       all hawks -> atlhawks
#       okc thunder-> okcthunder
#       breaks@out -> breaks out
#    WRONG USERNAMES (changed only in instagram_usernames.csv):
#       original00g -> youngheirgordon
#       bobimarjanovic13 -> boban
#       enikonhart -> enikohart
#       planetpat5 -> patconnaughton
#       steph -> stephencurry30
#       unclejg8 -> unclejeffgreen
#       frenchtouch20 -> 7tlc
#       shaigilalex -> shai
#       mitchellness -> mitchellandness
#       the2kferguson -> fergski23
#       kuzmakyle -> kuz
#       1jordanbell -> jbell
#       iisaiahthomas -> isaiahthomas
#       furkankorkkmaz -> furkankorkmaz
#       gdhayward -> gordonhayward
#    filled out account type- 1=Player (current/near current), 2=Team, 3=Celebrity, 4=Company/Organization, 5=Other,

df_train = pd.read_csv('text_processing/training_set_same.csv')
df_holdout = pd.read_csv('text_processing/holdout_set_same.csv')

df_train = df_train.loc[:,'Description']
df_holdout = df_holdout.loc[:,'Description']

users = set()

for x in df_train:
    if x is not None and type(x) == str:
        users.update(set(re.findall('@[A-Za-z0-9_.]+', x)))

for x in df_holdout:
    if x is not None and type(x) == str:
        users.update(set(re.findall('@[A-Za-z0-9_.]+', x)))

file = open('text_processing/instagram_usernames.csv', 'w')

file.write('scraped username,actual username,type\n')

for user in users:
    file.write(f'{user[1:]},{user[1:]},\n')

