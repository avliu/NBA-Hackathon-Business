import pandas as pd
import numpy as np
import requests
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 1. prepare text data- map scraped instagram usernames to actual usernames
#    WRONG USERNAMES:
#       all hawks -> atlhawks
#       okc thunder-> okcthunder
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
#    filled out account type- 1=Player (current/near current), 2=Team, 3=Celebrity, 4=Company/Organization

df_train = pd.read_csv('text_processing/training_set.csv')
df_holdout = pd.read_csv('text_processing/holdout_set.csv')

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


# 2 prepare text data- initialize a table to hold account user and corresponding number of followers,
# so that we don't have to hit the api for users we've already asked for

df_instagram_usernames = pd.read_csv('text_processing/instagram_usernames.csv', index_col=0)
index = list(df_instagram_usernames['actual username'])
index = [i.lower() for i in index]
index = set(index)
df_instagram_followers = pd.DataFrame(-1,index=index, columns=['followers'])
df_instagram_followers.to_csv('text_processing/instagram_followers.csv')


# 3. prepare text data- grab holdout and training set, add on 8 columns:
#   {number of accounts, number of total followers} for each type- 1,2,3,4
# METHODOLOGY:
#   always try to grab followers from instagram_followers_x.csv to avoid redundant api call
#   if necessary, call get_followers_count
#   when we get kicked out for too many api calls (response code 429), report it and write out progress
# HISTORY:
#   1. found stats for training_set
#       a. got through lines 1-806, output in training_set_stats_1.csv, instagram_followers_1.csv
#       b. 806-4556, training_set_stats_2.csv, instagram_followers_2.csv
#       c. 2541-4556, training_set_stats_3.csv, instagram_followers_3.csv
#       d. 2541-end, training_set_stats_4.csv, instagram_followers_4.csv
#   2. found stats for holdout_set
#       a. 1-end, holdout_set_stats_1.csv, instagram_followers_5.csv

def get_followers_count(username):
    ret = requests.get(f'https://www.instagram.com/{username}/?__a=1')
    if ret.status_code == 404:
        return 0
    elif ret.status_code == 429:
        return -429
    ret_dict = ret.json()
    count = ret_dict['graphql']['user']['edge_followed_by']['count']
    return count


dataset = 'training_set'
dataset_iteration = 4
followers_iteration = 5

df_instagram_usernames = pd.read_csv('text_processing/instagram_usernames.csv', index_col=0)
df_instagram_usernames.index = map(str.lower, df_instagram_usernames.index)
df_instagram_followers = pd.read_csv(f'text_processing/instagram_followers_{followers_iteration}.csv', index_col=0)

df_train = pd.read_csv(f'text_processing/{dataset}.csv')
df_train = df_train.loc[:,'Description']

columns = ['player', 'player_followers', 'team', 'team_followers', 'celebrity', 'celebrity_followers',
         'organization', 'organization_followers']
df_instagram_stats = pd.DataFrame(0, index=range(df_train.shape[0]), columns=columns)

stop_flag = False

for i in range(0, len(df_train)):
    x = df_train[i]
    if type(x)==str:
        scraped_usernames = re.findall('@[A-Za-z0-9_.]+', x)
        scraped_usernames = [s[1:].lower() for s in scraped_usernames]
        for scraped_username in scraped_usernames:
            actual_username = df_instagram_usernames.loc[scraped_username, 'actual username']
            user_type = df_instagram_usernames.loc[scraped_username, 'type']
            if type(actual_username) == pd.Series:
                actual_username = actual_username.iloc[0].lower()
                user_type = user_type.iloc[0]
            else:
                actual_username = actual_username.lower()
            follower_count = 0
            follower_count = df_instagram_followers.loc[actual_username, 'followers']
            if follower_count == -1:
                follower_count = get_followers_count(actual_username)
                if follower_count == -429:
                    stop_flag = True
                    break
                df_instagram_followers.loc[actual_username, 'followers'] = follower_count
            df_instagram_stats.iloc[i, (user_type - 1) * 2] += 1
            df_instagram_stats.iloc[i, (user_type - 1) * 2 + 1] += follower_count
        if stop_flag:
            print(f'kicked out at line {i}')
            break

df_instagram_followers.to_csv(f'text_processing/instagram_followers_{followers_iteration + 1}.csv')
df_instagram_stats.to_csv(f'text_processing/{dataset}_stats_{dataset_iteration + 1}.csv')


# 4. Since the stats were updated iteratively for training_set,
# we still need to combine all of them to create the final csv output
# put followers, holdout set stats in a final csv output for consistency

df_training_final = pd.read_csv('text_processing/training_set_stats_1.csv', index_col=0).loc[:805]
df_training_final = df_training_final.append(
    pd.read_csv('text_processing/training_set_stats_2.csv', index_col=0).loc[806:2540])
df_training_final = df_training_final.append(
    pd.read_csv('text_processing/training_set_stats_3.csv', index_col=0).loc[2541:4555])
df_training_final = df_training_final.append(
    pd.read_csv('text_processing/training_set_stats_4.csv', index_col=0).loc[4556:])
df_training_final.to_csv('text_processing/training_set_stats_final.csv')

df_holdout_final = pd.read_csv('text_processing/holdout_set_stats_1.csv', index_col=0)
df_holdout_final.to_csv('text_processing/holdout_set_stats_final.csv')

df_followers_final = pd.read_csv('text_processing/instagram_followers_6.csv', index_col=0)
df_followers_final.to_csv('text_processing/instagram_followers_final.csv')


# FINAL OUTPUTS: training_set_stats_final.csv, holdout_set_stats_final.csv, instagram_folowers_final.csv
