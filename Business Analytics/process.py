import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np


def build_model(input_size):
    input = keras.Input(shape=(input_size,))
    x = keras.layers.Dense(10, activation='sigmoid')(input)
    x = keras.layers.Dropout(.5)(x)
    x = keras.layers.Dense(1, activation='relu')(x)

    model = keras.Model(inputs=[input], outputs=[x])
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mape'])

    return model

train = pd.read_csv('training_set_final_0.csv')
train_features = train.iloc[:, 1:].to_numpy()
train_labels = train.iloc[:, 0].to_numpy()


num_epochs = 20
all_scores = []
k = 50
num_val_samples = train_features.shape[0] // k

for i in range(0, k):
    print(f'processing fold {k}')
    val_data = train_features[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i+1) * num_val_samples]

    partial_train_data = np.concatenate([train_features[:i * num_val_samples],
                                         train_features[(i+1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_labels[:i * num_val_samples],
                                         train_labels[(i+1) * num_val_samples:]], axis=0)
    model = build_model(train_features.shape[-1])
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1)
    val_mape = model.evaluate(val_data, val_targets)
    all_scores.append(val_mape)

a = 3