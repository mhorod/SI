import tensorflow as tf

from data import *
from common import *

LAYERS_TO_FREEZE = 100

def create_fine_tunable_model(model, mobilenet):
    # Unlock mobilenet and freeze the first 100 layers
    mobilenet.trainable = True
    for layer in mobilenet.layers[:LAYERS_TO_FREEZE]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

def main():
    transfer_learned = tf.keras.models.load_model('models/transfer_learning.model')
    mobilenet = transfer_learned.layers[2] # In transfer_learning.py, mobilenet is the 3rd layer

    model = create_fine_tunable_model(transfer_learned, mobilenet)
    history = train_and_save_model(model, 'fine_tuning', train_data, validation_data, 5)
    evaluate_model(model, 'fine_tuning', test_data)

if __name__ == '__main__':
  main()