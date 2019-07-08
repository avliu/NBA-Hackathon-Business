import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np


def build_model(start_node_size, optimizer, input_size_stats):

    input_stats = keras.Input(shape=(input_size_stats,))
    x = keras.layers.Dense(start_node_size, activation='elu')(input_stats)

    node_size = int(start_node_size/2)
    while node_size >= 8:
        x = keras.layers.Dense(node_size, activation='elu')(x)
        node_size = int(node_size/2)

    x = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=input_stats, outputs=x)

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

    train = pd.read_csv('training_set_final_2.csv')
    train = mix(train)
    train_features_stats = train.iloc[:, 1:].to_numpy()
    train_labels = train.iloc[:, 0].to_numpy()

    num_epochs = 30
    all_scores = []
    k = 5
    num_val_samples = train_features_stats.shape[0] // k

    for i in range(0, k):
        print(f'processing fold {i+1}')
        val_stats_data = train_features_stats[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_labels[i * num_val_samples: (i+1) * num_val_samples]

        partial_train_stats_data = np.concatenate([train_features_stats[:i * num_val_samples],
                                             train_features_stats[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_labels[:i * num_val_samples],
                                             train_labels[(i+1) * num_val_samples:]], axis=0)

        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        model = build_model(start_node_size, optimizer, train_features_stats.shape[-1])
        model.fit(partial_train_stats_data, partial_train_targets,
                  epochs=num_epochs, batch_size=1, callbacks=[early_stop], verbose=True)
        val_loss, val_mape = model.evaluate(val_stats_data, val_targets)
        all_scores.append(val_mape)

    print(f'start_node_size: {start_node_size}, optimizer: {optimizer}, val: {np.mean(all_scores)}')
    print('-------------------------------------------------------------------------------------')
    return model, val_mape

def final_train(start_node_size, optimizer):

    train = pd.read_csv('training_set_final_2.csv')
    train = mix(train)
    train_features_stats = train.iloc[:, 1:].to_numpy()
    train_labels = train.iloc[:, 0].to_numpy()

    num_epochs = 30

    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    model = build_model(start_node_size, optimizer, train_features_stats.shape[-1])
    model.fit(train_features_stats, train_labels,
              epochs=num_epochs, batch_size=1, callbacks=[early_stop], verbose=True)

    return model


best_model = None
best_mape = 100
best_node_size = 0
best_optimizer = ''

for i in (32, 64, 128):
    for j in ('adadelta', 'rmsprop', 'adam', 'adamax'):
        model, val_mape = train_and_test(i, j)
        if val_mape < best_mape:
            best_model = model
            best_mape = val_mape
            best_node_size = i
            best_optimizer = j

final_model = final_train(best_node_size, best_optimizer)

holdout_set = pd.read_csv('holdout_set_final_2.csv')
holdout_features_stats = holdout_set.iloc[:, 1:].to_numpy()
predictions = final_model.predict(x=holdout_features_stats)

df = pd.DataFrame(predictions, columns=['Engagements'])
df.to_csv('predictions_2.csv')

print(f'final model had node size {best_node_size} and optimizer {best_optimizer}')