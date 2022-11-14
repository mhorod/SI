import tensorflow as tf

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from data import *
from common import *

BASE_IMAGE_SIZE = (100, 100, 3)
IMAGE_SIZE = (224, 224, 3)

def create_model():
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE,
        include_top=False,
        weights='imagenet'
    )

    mobilenet.trainable = False

    model = Sequential([
        Input(shape=BASE_IMAGE_SIZE),
        Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        Rescaling(1./255),
        mobilenet,
        Flatten(),
        Dropout(0.5),
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
    history = train_and_save_model(model, 'transfer_learning', train_data, validation_data, 3)
    evaluate_model(model, 'transfer_learning', test_data)

if __name__ == '__main__':
  main()

