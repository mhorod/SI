import tensorflow as tf
import matplotlib.pyplot as plt

import pickle


def train_and_save_model(model, model_name, train_data, validation_data, epochs):
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs
    )

    model.save(f'models/{model_name}.model')

    with open(f'history/{model_name}_history.pickle', 'wb') as f:
        pickle.dump(history, f)

    return history


def evaluate_model(model, model_name, test_data):
    _, test_acc = model.evaluate(test_data)
    print("Model {model_name} accuracy: {:5.2f}%".format(
        test_acc * 100, model_name=model_name))


def load_and_plot_history(model_name):
    with open(f'history/{model_name}_history.pickle', 'rb') as f:
        plot_history(pickle.load(f))


def load_and_test(model_name, test_data):
    model = tf.keras.models.load_model(f'models/{model_name}.model')
    evaluate_model(model, model_name, test_data)


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.history['accuracy']))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
