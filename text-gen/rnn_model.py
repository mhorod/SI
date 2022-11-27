import random
import string

import tensorflow as tf

from text_model import TextModel

from prepare_dataset import *


class TextGenModel(tf.keras.Model):
    def __init__(self, base_model, lookup, temperature=1.0):
        super().__init__()
        self.base_model = base_model
        self._ids_from_chars = lookup.ids_from_chars
        self._chars_from_ids = lookup.chars_from_ids
        self.temperature = temperature

        self.vocab_size = len(lookup.ids_from_chars.get_vocabulary())

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices = skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[self.vocab_size])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def ids_from_chars(self, chars):
        return self._ids_from_chars(chars)

    @tf.function
    def chars_from_ids(self, ids):
        return self._chars_from_ids(ids)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Take input characters and convert into ids tensors
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model
        predicted_logits, states = self.base_model(inputs=input_ids, states=states,
                                            return_state=True)

        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature

        # Apply previously generated mask to prevent bad characters
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states



class BaseModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        embedded = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(embedded)

        output, states = self.gru(embedded, initial_state=states, training=training)
        output = self.dense(output, training=training)

        return (output, states) if return_state else output

class RnnModel(TextModel):
    def __init__(self, text_gen_model, base_model):
        self.base_model = base_model
        self.text_gen_model = text_gen_model

    @staticmethod
    def from_file(model_path):
        base_model = tf.saved_model.load(model_path)
        text_gen_model = tf.saved_model.load("one_step_model")
        return RnnModel(text_gen_model, base_model)

    def generate_text(self, length):
        prompt = random.choice(string.ascii_letters)
        return self.generate_from_prompt(prompt, length)

    def generate_text_from_prompt(self, prompt, length):
        next_char = tf.constant([prompt])
        result = [prompt]
        states = None
        while len(result) < length:
            next_char, states = self.text_gen_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)
        return result[0].numpy().decode('utf-8')

    def generate(self, length):
        return self.generate_text(length)
    
    def generate_from_prompt(self, prompt, length):
        return self.generate_text_from_prompt(prompt, length)

    def perplexity(self, text):
        return -1