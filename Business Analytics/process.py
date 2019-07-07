import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from preprocess_text import get_tokenizer, get_tokenized_text, get_text_embedding_matrix



def build_model(start_node_size, optimizer, input_size_stats, max_words, embedding_dim, maxlen, embedding_matrix):

    input_stats = keras.Input(shape=(input_size_stats,))

    input_text = keras.Input(shape=(maxlen,))
    embedding_layer = keras.layers.Embedding(max_words, embedding_dim, input_length=maxlen, name='embedding')(input_text)
    embedding_layer = keras.layers.Flatten()(embedding_layer)

    x = keras.layers.concatenate([input_stats, embedding_layer])

    node_size = start_node_size
    while node_size >= 8:
        x = keras.layers.Dense(node_size, activation='elu')(x)
        node_size = int(node_size/2)

    x = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=[input_stats, input_text], outputs=x)

    model.get_layer('embedding').set_weights([embedding_matrix])
    model.get_layer('embedding').trainable = False

    model.compile(optimizer=optimizer, loss='mse', metrics=['mape'])

    return model


def mix(df):
    return df.sample(frac=1).reset_index(drop=True)


def combine(df1, df2):
    new_df = df1.copy()
    for col in df2.columns:
        new_df[col] = df2.loc[:,col]
    return new_df


def train_and_test(start_node_size, optimizer):

    train = combine(pd.read_csv('training_set_final_3.csv'), pd.read_csv('text_processing/training_set_same_text.csv', index_col=0))
    train = mix(train)
    train_features_stats = train.iloc[:, 1:-1].to_numpy()
    train_features_text = train.iloc[:, -1]
    train_labels = train.iloc[:, 0].to_numpy()

    num_epochs = 30
    all_scores = []
    k = 5
    num_val_samples = train_features_stats.shape[0] // k

    max_words = 50  # We will only consider the top 10,000 words in the dataset
    maxlen = 50  # We will cut reviews after 100 words
    embedding_dim = 50

    tokenizer = get_tokenizer(max_words)
    embedding_matrix = get_text_embedding_matrix(tokenizer, embedding_dim, max_words)

    for i in range(0, k):
        print(f'processing fold {i+1}')
        val_stats_data = train_features_stats[i * num_val_samples: (i + 1) * num_val_samples]
        val_stats_text = train_features_text[i * num_val_samples: (i + 1) * num_val_samples]
        val_stats_text = get_tokenized_text(tokenizer, val_stats_text, maxlen)
        val_targets = train_labels[i * num_val_samples: (i+1) * num_val_samples]

        partial_train_stats_data = np.concatenate([train_features_stats[:i * num_val_samples],
                                             train_features_stats[(i + 1) * num_val_samples:]], axis=0)
        partial_train_text_data = np.concatenate([train_features_text[:i * num_val_samples],
                                             train_features_text[(i + 1) * num_val_samples:]], axis=0)
        partial_train_text_data = get_tokenized_text(tokenizer, partial_train_text_data, maxlen)
        partial_train_targets = np.concatenate([train_labels[:i * num_val_samples],
                                             train_labels[(i+1) * num_val_samples:]], axis=0)

        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        model = build_model(start_node_size, optimizer, train_features_stats.shape[-1], max_words, embedding_dim, maxlen, embedding_matrix)
        model.fit([partial_train_stats_data, partial_train_text_data], partial_train_targets,
                  epochs=num_epochs, batch_size=1, callbacks=[early_stop], verbose=False)
        val_loss, val_mape = model.evaluate([val_stats_data, val_stats_text], val_targets)
        all_scores.append(val_mape)

    print(f'start_node_size: {start_node_size}, optimizer: {optimizer}, val: {np.mean(all_scores)}')
    print('-------------------------------------------------------------------------------------')


for i in (16, 32, 64, 128, 256):
    for j in ('rmsprop', 'adadelta', 'adagrad', 'adam', 'adamax'):
        try:
            train_and_test(i, j)
        except:
            pass
