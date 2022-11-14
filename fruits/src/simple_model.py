import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

import matplotlib.pyplot as plt

import pickle

from data import *
from common import *


def create_model():
    model = tf.keras.models.Sequential([
        # Normalize data
        Rescaling(1./255, input_shape=(100, 100, 3)),

        # Augment data
        RandomFlip('horizontal'),

        # Convolutional layers
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),

        Dropout(0.5),

        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),

        Dropout(0.5),

        # Flatten the results to feed into a DNN
        Flatten(),
        Dense(128, activation="relu"),
        Dense(NUMBER_OF_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def main():
    model = create_model()
    history = train_and_save_model(
        model, 'simple_model', train_data, validation_data, 10)
    evaluate_model(model, 'simple_model', test_data)


if __name__ == '__main__':
    main()
