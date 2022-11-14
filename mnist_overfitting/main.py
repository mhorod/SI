import tensorflow as tf

from tensorflow.keras import datasets, layers, models 

import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np


(train_data, test_data), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

BATCH_SIZE = 128

# Limit dataset drastically to present overfitting
train_data = train_data.take(1000)


train_data = train_data.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_data = train_data.cache()
train_data = train_data.shuffle(ds_info.splits['train'].num_examples)
train_data = train_data.batch(BATCH_SIZE)
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

test_data = test_data.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_data = test_data.batch(BATCH_SIZE)
test_data = test_data.cache()
test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

def overfitting_model():
    return models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),


        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

def dropout_model():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.MaxPooling2D((2, 2)),
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

def regularized_model():
    return tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(10, activation='softmax')
    ])


def regularized_l1_model():
    return tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01)),
        layers.Dense(10, activation='softmax')
    ])



def batch_norm_model():
    return tf.keras.models.Sequential([

        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        # layers.BatchNormalization(),

        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

def all_model():
    return tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.MaxPooling2D((2, 2)),
        # layers.BatchNormalization(),

        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])


models = [
    ('overfitting', overfitting_model()),
    ('dropout 0.5', dropout_model()),
    ('regularized L1', regularized_l1_model()),
    ('regularized L2', regularized_model()),
    ("batch normalization", batch_norm_model()),
    ("dropout + L2 + norm", all_model())
]

TESTS = 10

histories = []

for name, model in models:
    training_accuracies = []
    test_accuracies = []
    for _ in range(TESTS): 
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )


        history = model.fit(train_data, epochs=25, validation_data=test_data)

        _, test_acc = model.evaluate(test_data)

        training_accuracies.append((history.history['accuracy'], history.history['val_accuracy']))
        test_accuracies.append(test_acc)

    average_test_accuracy = sum(test_accuracies) / len(test_accuracies)

    average_training_accuracies = np.mean(training_accuracies, axis=0)
    histories.append((name, average_training_accuracies, average_test_accuracy))



# Plot the loss and accuracy curves for training and validation
fig, axs = plt.subplots(1, len(histories), figsize=(15, 5))
for (name, (train_acc, val_acc), test_acc), ax in zip(histories, axs):
    # Plot accuracy
    ax.plot(train_acc, color='b', label="Training accuracy")
    ax.plot(val_acc, color='r', label="validation accuracy",axes =ax)

    # Plot test accuracy as a line
    ax.axhline(y=test_acc, color='g', linestyle='-', label="Test accuracy")
    ax.set_ylim([0, 1.5])
    ax.set_title(name)
    ax.legend(loc='best', shadow=True)

plt.show()