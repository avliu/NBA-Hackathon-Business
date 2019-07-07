import pandas as pd
import numpy as np
import requests
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 1. prepare text data- map scraped instagram usernames to actual usernames
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

# df_train = pd.read_csv('text_processing/training_set_same.csv')
# df_holdout = pd.read_csv('text_processing/holdout_set_same.csv')
#
# df_train = df_train.loc[:,'Description']
# df_holdout = df_holdout.loc[:,'Description']
#
# users = set()
#
# for x in df_train:
#     if x is not None and type(x) == str:
#         users.update(set(re.findall('@[A-Za-z0-9_.]+', x)))
#
# for x in df_holdout:
#     if x is not None and type(x) == str:
#         users.update(set(re.findall('@[A-Za-z0-9_.]+', x)))
#
# file = open('text_processing/instagram_usernames.csv', 'w')
#
# file.write('scraped username,actual username,type\n')
#
# for user in users:
#     file.write(f'{user[1:]},{user[1:]},\n')


# 2 prepare text data- initialize a table to hold account user -> followers,
# so that we don't have to hit the api for users we've already asked for

# df_instagram_usernames = pd.read_csv('text_processing/instagram_usernames.csv', index_col=0)
# index = list(df_instagram_usernames['actual username'])
# index = [i.lower() for i in index]
# index = set(index)
# df_instagram_followers = pd.DataFrame(-1,index=index, columns=['followers'])
# df_instagram_followers.to_csv('text_processing/instagram_followers.csv')


# 3. prepare text data- grab holdout and training set, add on 8 columns:
# {number of accounts, number of total followers} for each type- 1,2,3,4
# METHODOLOGY:
#   always try to grab followers from instagram_followers_x.csv to avoid redundant api call
#   if necessary, then call get_followers_count
#   when we get kicked out for too many api calls (response code 429), report it and write out progress
# HISTORY:
#   1. found stats for training_set
#       a. got through lines 1-806, output in training_set_same_stats_1.csv, instagram_followers_1.csv
#       b. 806-4556, training_set_same_stats_2.csv, instagram_followers_2.csv
#       c. 2541-4556, training_set_same_stats_3.csv, instagram_followers_3.csv
#       d. 2541-end, training_set_same_stats_4.csv, instagram_followers_4.csv
#           (accidentally forgot to update range for i, so it accidentally re-fetched 2541-4556
#   2. found stats for hodlout_set
#       a. 1-end, holdout_set_same_stats_1.csv, instagram_followers_5.csv

# def get_followers_count(username):
#     ret = requests.get(f'https://www.instagram.com/{username}/?__a=1')
#     if ret.status_code == 404:
#         return 0
#     elif ret.status_code == 429:
#         return -429
#     ret_dict = ret.json()
#     count = ret_dict['graphql']['user']['edge_followed_by']['count']
#     return count
#
# dataset = 'training_set_same'
#
# df_instagram_usernames = pd.read_csv('text_processing/instagram_usernames.csv', index_col=0)
# df_instagram_usernames.index = map(str.lower, df_instagram_usernames.index)
# df_instagram_followers = pd.read_csv('text_processing/instagram_followers_5.csv', index_col=0)
#
# df_train = pd.read_csv(f'text_processing/{dataset}.csv')
# df_train = df_train.loc[:,'Description']
#
# columns = ['player', 'player_followers', 'team', 'team_followers', 'celebrity', 'celebrity_followers',
#          'organization', 'organization_followers']
# df_instagram_stats = pd.DataFrame(0, index=range(df_train.shape[0]), columns=columns)
#
# stop_flag = False
#
# for i in range(0, len(df_train)):
#     x = df_train[i]
#     if type(x)==str:
#         scraped_usernames = re.findall('@[A-Za-z0-9_.]+', x)
#         scraped_usernames = [s[1:].lower() for s in scraped_usernames]
#         for scraped_username in scraped_usernames:
#             actual_username = df_instagram_usernames.loc[scraped_username, 'actual username']
#             user_type = df_instagram_usernames.loc[scraped_username, 'type']
#             if type(actual_username) == pd.Series:
#                 actual_username = actual_username.iloc[0].lower()
#                 user_type = user_type.iloc[0]
#             else:
#                 actual_username = actual_username.lower()
#             follower_count = 0
#             follower_count = df_instagram_followers.loc[actual_username, 'followers']
#             if follower_count == -1:
#                 follower_count = get_followers_count(actual_username)
#                 if follower_count == -429:
#                     stop_flag = True
#                     break
#                 df_instagram_followers.loc[actual_username, 'followers'] = follower_count
#             df_instagram_stats.iloc[i, (user_type - 1) * 2] += 1
#             df_instagram_stats.iloc[i, (user_type - 1) * 2 + 1] += follower_count
#         if stop_flag:
#             print(f'kicked out at line {i}')
#             break
#
# df_instagram_followers.to_csv('text_processing/instagram_followers_6.csv')
# df_instagram_stats.to_csv(f'text_processing/{dataset}_stats_1_1.csv')


# 4. Since the stats were updated iteratively for training_set,
# we still need to combine all of them to create the final text output

# df_training_final = pd.read_csv('text_processing/training_set_same_stats_1.csv', index_col=0).loc[:805]
# df_training_final = df_training_final.append(
#     pd.read_csv('text_processing/training_set_same_stats_2.csv', index_col=0).loc[806:2540])
# df_training_final = df_training_final.append(
#     pd.read_csv('text_processing/training_set_same_stats_3.csv', index_col=0).loc[2541:4555])
# df_training_final = df_training_final.append(
#     pd.read_csv('text_processing/training_set_same_stats_4.csv', index_col=0).loc[4556:])
# df_training_final.to_csv('text_processing/training_set_stats_final.csv')
#
# # df_holdout_final = pd.read_csv('text_processing/holdout_set_same_stats_1.csv', index_col=0)
# # df_holdout_final.to_csv('text_processing/holdout_set_same_stats_final.csv')
#
# df_followers_final = pd.read_csv('text_processing/instagram_followers_6.csv', index_col=0)
# df_followers_final.to_csv('text_processing/instagram_followers_final.csv')


# FINAL OUTPUTS: training_set_stats_final.csv, holdout_set_same_stats_final.csv, instagram_folowers_final.csv


# 5. Actual text processing- creating file for just text, then tokenizing it

# df_train = pd.read_csv('text_processing/training_set_same.csv')
# df_train = pd.DataFrame(df_train.loc[:,'Description'])
# df_train.to_csv('text_processing/training_set_same_text.csv', index='Description')
#
# df_holdout = pd.read_csv('text_processing/holdout_set_same.csv')
# df_holdout = pd.DataFrame(df_holdout.loc[:,'Description'])
# df_holdout.to_csv('text_processing/holdout_set_same_text.csv', index='Description')


def get_username_tokenizer(max_usernames=200):
    texts = pd.read_csv(f'text_processing/training_set_same_text.csv', index_col=0)
    texts = texts.loc[:, 'Description'].to_list()
    new_texts = []
    for text in texts:
        text = str(text)
        usernames = set(re.findall('@[A-Za-z0-9_.]+', text))
        usernames = usernames.union(set(re.findall('#[A-Za-z0-9]+', text)))
        text = ''
        for username in usernames:
            text += username
            text += ' '
        new_texts.append(text)
    tokenizer = Tokenizer(num_words=max_usernames, filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(new_texts)

    return tokenizer


def get_tokenizer(max_words):

    texts = pd.read_csv(f'text_processing/training_set_same_text.csv', index_col=0)
    texts = texts.loc[:, 'Description'].to_list()
    texts = [str(text) for text in texts]

    tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(texts)

    return tokenizer


def get_tokenized_text(tokenizer, texts, maxlen):

    texts = [str(text) for text in texts]

    sequences = tokenizer.texts_to_sequences(texts)

    data = pad_sequences(sequences, maxlen=maxlen)

    return data


def get_text_embedding_matrix(tokenizer, embedding_dim, max_words):

    embeddings_index = {}
    f = open(f'glove.6B/glove.6B.50d.txt', 'r', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

    return embedding_matrix

