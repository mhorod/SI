import tensorflow as tf

from prepare_dataset import *

import numpy as np
import matplotlib.pyplot as plt

class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, rnn_units):
        super().__init__()
        self.text_processor = text_processor
        self.rnn_units = rnn_units
        self.vocab_size = text_processor.vocabulary_size()

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, rnn_units, mask_zero=True)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        x, state = self.gru(x)
        return x, state

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        return self(self.text_processor(texts).to_tensor())[0]


class Decoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, rnn_units):
        super().__init__()
        self.text_processor = text_processor
        self.rnn_units = rnn_units
        self.vocab_size = text_processor.vocabulary_size()

        self.word_to_index = tf.keras.layers.StringLookup(vocabulary=self.text_processor.get_vocabulary(), mask_token='')
        self.index_to_word = tf.keras.layers.StringLookup(vocabulary=self.text_processor.get_vocabulary(), mask_token='', invert=True)

        self.start_token = self.word_to_index('<bos>')
        self.end_token = self.word_to_index('<eos>')

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, rnn_units, mask_zero=True)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs, states):
        x = self.embedding(inputs)
        x, state = self.gru(x, initial_state=states)
        x = self.dense(x)
        return x, state

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(tokens)
        state = self.gru.get_initial_state(embedded)[0]
        return tokens, state, done

    def tokens_to_text(self, tokens):
        words =  self.index_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return result.numpy()

    def get_next_token(self, next_token, state, done):
        logits, state = self(next_token, state)
        next_token = tf.argmax(logits, axis=-1)
        done = done | (next_token == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
        return next_token, state, done

class Seq2Seq(tf.keras.Model):
    def __init__(self, context_text_processor, target_text_processor, rnn_units):
        super().__init__()
        self.encoder = Encoder(context_text_processor, rnn_units)
        self.decoder = Decoder(target_text_processor, rnn_units)

    def call(self, inputs):
        context, target = inputs
        _, encoder_states = self.encoder(context)
        decoded, _ = self.decoder(target, encoder_states)
        return decoded

    def translate(self, texts):
        context = self.encoder.convert_input(texts)
        tokens = []
        next_token, state, done = self.decoder.get_initial_state(context)

        for _ in range(10):
            next_token, state, done = self.decoder.get_next_token(next_token, state, done)
            tokens.append(next_token)

        tokens = tf.concat(tokens, axis=1)
        result = self.decoder.tokens_to_text(tokens)
        return result


def masked_loss(y_true, y_pred):
    # Because we pad tokens with 0, we have to calculate the loss only on the non-zero tokens
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(y_true != 0, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    # Same thing for accuracy
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), dtype=y_true.dtype)
    matched = tf.cast(tf.equal(y_true, y_pred), dtype=y_true.dtype)
    mask = tf.cast(y_true != 0, dtype=matched.dtype)
    return tf.reduce_sum(matched) / tf.reduce_sum(mask)



MAX_VOCAB_SIZE = 10000

translations = load_translations_from_file("spa-small.txt")
train_ds, test_ds = datasets_from_translations(translations)

context_text_preprocessor, target_text_preprocessor = make_text_preprocessors(MAX_VOCAB_SIZE, train_ds)
train_ds = process_dataset(train_ds, context_text_preprocessor, target_text_preprocessor)

RNN_UNITS = 2048
model = Seq2Seq(context_text_preprocessor, target_text_preprocessor, RNN_UNITS)

model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_accuracy, masked_loss])
model.fit(train_ds, epochs=200)

sequences = ["Hello", "Run"]
translated = model.translate(sequences)
for sequence, translation in zip(sequences, translated):
    print(sequence, '->', translation.decode('utf-8'))